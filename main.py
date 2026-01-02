import yfinance as yf
import pandas as pd
import pandas_ta as ta
import twstock
from tqdm import tqdm
from FinMind.data import DataLoader
from textblob import TextBlob
import feedparser
import time
import sys
import datetime
import requests
import os
import json
import random
import time
import logging
from google import genai
from dotenv import load_dotenv

# Setup Logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, 'bot_activity.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Load Environment Variables from .env in BASE_DIR
load_dotenv(os.path.join(BASE_DIR, '.env'))

logging.info(f"🚀 Bot starting in Production Mode. Watched Directory: {BASE_DIR}")

# Constants
INITIAL_CAPITAL = 100000
TARGET_MONTHLY_RETURN = 0.20
TEST_MODE = False
INTRADAY_MODE = True
MAX_POSITION_RATIO = 0.40
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio.json")
SAFE_ASSET_TICKER = "00878.TW"

# Global State for Market Sentiment
DAILY_MARKET_SENTIMENT = "Neutral"

# Battlefield Targets (High Volume / Momentum)
BATTLEFIELD_TARGETS = [
    # Semis & Tech (The Core)
    "2330.TW", "2454.TW", "2317.TW", "2308.TW", "2303.TW", "3711.TW", "3034.TW", "3037.TW", "2379.TW", "3008.TW",
    # AI Server & Hardware (High Momentum)
    "3231.TW", "2382.TW", "6669.TW", "2356.TW", "2376.TW", "2357.TW", "2301.TW", "2324.TW", "2421.TW", "2368.TW",
    # Power & Energy (Policy Plays)
    "1513.TW", "1519.TW", "1503.TW", "1504.TW", "1609.TW", "6806.TW",
    # Shipping (Volatility Kings)
    "2603.TW", "2609.TW", "2615.TW", "2618.TW", "2610.TW", "2606.TW",
    # Financials (Stability)
    "2881.TW", "2882.TW", "2891.TW", "2886.TW", "2884.TW", "2892.TW", "2885.TW", "5880.TW", "2880.TW", "2890.TW",
    # High Volume / Popular / ETFs (Proxies)
    "0050.TW", "0056.TW", "00929.TW", "00919.TW",
    "2344.TW", "2409.TW", "3481.TW", "2002.TW", "1101.TW", "2353.TW", "2327.TW", "2449.TW",
    "3017.TW", "3035.TW", "3044.TW", "2383.TW", "2363.TW", "2337.TW", "2492.TW", "3019.TW", "2408.TW"
]

# De-duplicate just in case
BATTLEFIELD_TARGETS = list(set(BATTLEFIELD_TARGETS))

# Validate Keys
if not DISCORD_WEBHOOK_URL:
    logging.error("Missing Environment Variable: DISCORD_WEBHOOK_URL.")
    sys.exit(1)
if not GEMINI_API_KEY:
    logging.error("Missing Environment Variable: GEMINI_API_KEY.")
    sys.exit(1)

def get_stock_name_cn(ticker):
    """
    Attempts to fetch the Chinese name of the stock using twstock.
    Expects ticker format like '2330.TW' or '2330.TWO'.
    Returns the Chinese name or the original ticker if not found.
    """
    try:
        clean_code = ticker.split('.')[0]
        if clean_code in twstock.codes:
             return twstock.codes[clean_code].name
        return ticker
    except:
        return ticker

def with_retry(retries=3, delay=10):
    """
    Decorator for retry logic with exponential backoff.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    wait = delay * (2 ** i)
                    print(f"Error in {func.__name__}: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
            print(f"Failed to execute {func.__name__} after {retries} retries.")
            return None
        return wrapper
    return decorator

@with_retry()
def fetch_macro_news():
    """
    Fetches macro news from Anue RSS (https://news.cnyes.com/rss/cat/209).
    Returns a string summary of the top 5 news items.
    """
    try:
        url = "https://news.cnyes.com/rss/cat/209"
        feed = feedparser.parse(url)

        summary_list = []
        # Get top 5 entries
        for entry in feed.entries[:5]:
            title = entry.title
            # summary is often HTML, strip tags if needed or just use title
            # Anue summary is usually clean enough or just take title
            summary_list.append(f"- {title}")

        if not summary_list:
            return "No Macro News Available."

        return "\n".join(summary_list)
    except Exception as e:
        logging.error(f"Error fetching macro news: {e}")
        return "Error fetching macro news."

@with_retry()
def fetch_stock_news(ticker):
    """
    Fetches the latest specific news for a stock using yfinance.
    Returns the top 2 headlines with links.
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return "No specific news found."

        news_items = []
        for item in news[:2]: # Top 2
            title = item.get('title', 'No Title')
            link = item.get('link', '#')
            news_items.append(f"- {title} ({link})")

        return "\n".join(news_items)
    except Exception as e:
        logging.error(f"Error fetching news for {ticker}: {e}")
        return "Error fetching stock news."

def calculate_transaction_fee(amount):
    """
    Calculates transaction fee: 0.1425%, min 20 TWD.
    """
    return max(20, int(amount * 0.001425))

def calculate_tax(amount):
    """
    Calculates transaction tax: 0.3%.
    """
    return int(amount * 0.003)

@with_retry()
def get_realtime_price(ticker):
    """
    Fetches the latest trade price from twstock realtime API.
    Returns float price or None if failed.
    """
    try:
        clean_code = ticker.split('.')[0]
        realtime_data = twstock.realtime.get(clean_code)

        if realtime_data['success']:
            price_str = realtime_data['realtime']['latest_trade_price']
            # Sometimes price is '-' if no trades yet, fallback to best bid/ask or previous close
            if price_str == '-':
                 # Try open
                 price_str = realtime_data['realtime'].get('open', '-')

            if price_str != '-':
                return float(price_str)

        return None
    except Exception as e:
        logging.error(f"Error fetching realtime price for {ticker}: {e}")
        return None

def load_portfolio():
    """
    Loads portfolio from JSON file or creates a default one if not exists.
    Performs migration to rich format and segments Active/Safe holdings.
    """
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                portfolio = json.load(f)

            migrated = False

            # Migration 1: 'holdings' -> 'active_holdings'
            if 'holdings' in portfolio and 'active_holdings' not in portfolio:
                logging.info("Migrating 'holdings' to 'active_holdings'...")
                portfolio['active_holdings'] = portfolio['holdings']
                del portfolio['holdings']
                portfolio['safe_holdings'] = {}
                portfolio['initial_principal'] = INITIAL_CAPITAL
                migrated = True

            # Initialize Safe Holdings if missing
            if 'safe_holdings' not in portfolio:
                portfolio['safe_holdings'] = {}
                migrated = True

            # Initialize Initial Principal if missing
            if 'initial_principal' not in portfolio:
                portfolio['initial_principal'] = INITIAL_CAPITAL
                migrated = True

            # Data Integrity Check on Active Holdings
            for ticker, data in portfolio['active_holdings'].items():
                if isinstance(data, int):
                    logging.info(f"Migrating {ticker} to rich format...")
                    realtime_price = get_realtime_price(ticker)
                    cost_basis = realtime_price if realtime_price else 0.0

                    portfolio['active_holdings'][ticker] = {
                        "shares": data,
                        "cost": cost_basis,
                        "buy_reason": "Legacy Position"
                    }
                    migrated = True
                elif isinstance(data, dict) and "buy_reason" not in data:
                    logging.info(f"Migrating {ticker} adding buy_reason...")
                    portfolio['active_holdings'][ticker]["buy_reason"] = "Legacy Position"
                    migrated = True

            if migrated:
                logging.info("Portfolio migration complete. Saving...")
                save_portfolio(portfolio)

            return portfolio

        except json.JSONDecodeError:
            logging.error("Error decoding portfolio.json, creating new.")

    # Default Portfolio
    return {
        "balance": INITIAL_CAPITAL,
        "initial_principal": INITIAL_CAPITAL,
        "active_holdings": {},
        "safe_holdings": {},
        "history": []
    }

def get_recent_performance_summary(portfolio, limit=5):
    """
    Generates a summary string of recent closed trades for AI context.
    """
    summary = []
    closed_trades = [
        log for log in portfolio['history']
        if log['action'] in ["SELL", "CLEAR", "REDUCE"]
    ]

    # Get last N trades (reverse order)
    recent = closed_trades[-limit:]

    for i, trade in enumerate(reversed(recent), 1):
        ticker = trade.get('ticker', 'Unknown')
        pnl = trade.get('pnl_percentage', 0.0)
        buy_reason = trade.get('buy_reason', 'N/A')
        sell_reason = trade.get('reason', 'N/A')

        result = "WIN" if pnl > 0 else "LOSS"

        # Extract rich indicators with defaults for legacy data
        inds = trade.get('indicators', {})
        rsi = inds.get('RSI', 'N/A')
        macd = inds.get('MACD_Hist', 'N/A')
        bb = inds.get('BB_Pct', 'N/A')
        ma = inds.get('MA_Trend', 'N/A')

        summary.append(
            f"• {ticker} ({result} {pnl:+.2f}%): {buy_reason} -> {sell_reason} | Ind: RSI={rsi}, MACD={macd}, BB%={bb}, MA={ma}"
        )

    if not summary:
        return "No recent closed trades."

    return "\n".join(summary)

def save_portfolio(portfolio):
    """
    Saves portfolio data to JSON file.
    """
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=4)

def get_segment_values(portfolio):
    """
    Calculates the Market Value of Active and Safe segments.
    Returns (active_total, safe_total, active_val, safe_val)
    active_total = Cash + Active Holdings Value
    safe_total = Safe Holdings Value
    """
    active_val = 0
    safe_val = 0

    # Calculate Active Holdings Value
    for ticker, data in portfolio['active_holdings'].items():
        shares = data['shares']
        price = get_realtime_price(ticker)
        if not price:
            # Fallback
            try:
                price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
            except:
                price = 0
        active_val += shares * price

    # Calculate Safe Holdings Value
    for ticker, data in portfolio['safe_holdings'].items():
        shares = data['shares']
        price = get_realtime_price(ticker)
        if not price:
            try:
                price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
            except:
                price = 0
        safe_val += shares * price

    active_total = portfolio['balance'] + active_val
    safe_total = safe_val

    return active_total, safe_total, active_val, safe_val

def check_capital_rescue(portfolio):
    """
    Function A: Safety Net (Loss Recovery).
    Trigger: After SELL trades AND at Market Close.
    Logic: If Active < 100k AND Safe > 0, sell Safe Asset to restore Principal.
    """
    logging.info("Checking for Capital Rescue (Safety Net)...")
    active_total, safe_total, _, _ = get_segment_values(portfolio)
    initial_principal = portfolio.get('initial_principal', INITIAL_CAPITAL)

    triggered = False

    if active_total < initial_principal and safe_total > 0:
        deficit = initial_principal - active_total
        # Threshold to avoid micro-trades
        if deficit > 1000:
            price = get_realtime_price(SAFE_ASSET_TICKER)
            if price:
                amount_to_liquidate = min(deficit, safe_total)
                shares_to_sell = int(amount_to_liquidate / price)

                current_shares = portfolio['safe_holdings'].get(SAFE_ASSET_TICKER, {}).get('shares', 0)
                shares_to_sell = min(shares_to_sell, current_shares)

                if shares_to_sell > 0:
                    gross_rev = shares_to_sell * price
                    fee = calculate_transaction_fee(gross_rev)
                    tax = calculate_tax(gross_rev)
                    net_rev = gross_rev - fee - tax

                    portfolio['balance'] += net_rev
                    portfolio['safe_holdings'][SAFE_ASSET_TICKER]['shares'] -= shares_to_sell

                    # Cleanup if 0
                    if portfolio['safe_holdings'][SAFE_ASSET_TICKER]['shares'] == 0:
                        del portfolio['safe_holdings'][SAFE_ASSET_TICKER]

                    msg = f"⛑️ **Rescue Triggered**: Sold ${net_rev:,.0f} of Safe Asset to restore Principal."
                    logging.info(msg)
                    send_discord_notification(msg)
                    triggered = True

    return portfolio, triggered

def check_profit_harvest_ai(portfolio):
    """
    Function B: AI Profit Harvesting (The Manager).
    Trigger: ONLY at Market Close.
    Logic: If Active > 100k, ask Gemini: KEEP or HARVEST?
    """
    logging.info("Checking for AI Profit Harvesting...")
    active_total, safe_total, _, _ = get_segment_values(portfolio)
    initial_principal = portfolio.get('initial_principal', INITIAL_CAPITAL)

    surplus = active_total - initial_principal

    if surplus <= 0:
        return portfolio

    # AI Decision Logic
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        active_positions_count = len(portfolio['active_holdings'])
        current_cash = portfolio['balance']

        prompt = (
            f"We have a PROFIT SURPLUS of ${surplus:,.0f} (Active Total: ${active_total:,.0f}).\n"
            f"Current Context:\n"
            f"- Active Cash: ${current_cash:,.0f}\n"
            f"- Active Positions Count: {active_positions_count}\n"
            f"- Market Sentiment: {DAILY_MARKET_SENTIMENT}\n"
            f"Strategy: Aggressive 20% Monthly Return.\n\n"
            f"Decision: Should we (A) KEEP this cash to buy more aggressive stocks tomorrow? or (B) HARVEST profit to Safe Bucket ({SAFE_ASSET_TICKER})?\n"
            f"Output JSON: {{'action': 'KEEP' or 'HARVEST', 'amount': <float>}}.\n"
            f"Note: 'amount' is how much of the surplus to move (if HARVEST). Keep it 0 if KEEP."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        decision = {}
        if response.text:
            decision = parse_ai_json(response.text)

        action = decision.get('action', 'KEEP')
        amount = float(decision.get('amount', 0.0))

        if action == "HARVEST" and amount > 0:
            # Execution: Buy 00878
            # Limit by actual cash available and surplus
            real_amount = min(amount, current_cash)
            # Also limit by surplus? The prompt implies amount <= surplus, but strict math:
            # We are effectively moving cash.

            if real_amount > 1000: # Min threshold
                price = get_realtime_price(SAFE_ASSET_TICKER)
                if price:
                     if real_amount > price:
                        shares_to_buy = int(real_amount / price)
                        if shares_to_buy > 0:
                            gross_cost = shares_to_buy * price
                            fee = calculate_transaction_fee(gross_cost)
                            total_cost = gross_cost + fee

                            if portfolio['balance'] >= total_cost:
                                portfolio['balance'] -= total_cost

                                # Update Safe Holdings
                                current_data = portfolio['safe_holdings'].get(SAFE_ASSET_TICKER, {'shares': 0, 'cost': 0.0})
                                old_shares = current_data['shares']
                                old_cost = current_data['cost'] * old_shares

                                new_shares = old_shares + shares_to_buy
                                new_avg = (old_cost + total_cost) / new_shares

                                portfolio['safe_holdings'][SAFE_ASSET_TICKER] = {
                                    "shares": new_shares,
                                    "cost": new_avg,
                                    "buy_reason": "AI Harvest"
                                }

                                msg = f"💰 **AI Harvested**: ${total_cost:,.0f} moved to Safe Bucket."
                                logging.info(msg)
                                send_discord_notification(msg)
        else:
            logging.info(f"🔥 AI decided to reinvest profit (KEEP). Surplus ${surplus:,.0f} remains in Active Cash.")

    except Exception as e:
        logging.error(f"AI Harvesting Error: {e}")

    return portfolio

def calculate_portfolio_value(portfolio):
    """
    Calculates total portfolio value (Cash + Active + Safe).
    Returns (total_value, profit_loss_pct, details_str).
    """
    active_total, safe_total, active_val, safe_val = get_segment_values(portfolio)
    total_value = active_total + safe_total - portfolio['balance'] # Avoid double counting cash?
    # Wait, active_total = Cash + Active_Val.
    # Safe_total = Safe_Val.
    # Total Equity = Cash + Active_Val + Safe_Val.
    # So Total = active_total + safe_total. Correct.
    total_value = portfolio['balance'] + active_val + safe_val

    initial_principal = portfolio.get('initial_principal', INITIAL_CAPITAL)
    pl_pct = ((total_value - initial_principal) / initial_principal) * 100

    details = []
    # Active Details
    for ticker, data in portfolio['active_holdings'].items():
        details.append(f"{ticker}(A):{data['shares']}")
    # Safe Details
    for ticker, data in portfolio['safe_holdings'].items():
        details.append(f"{ticker}(S):{data['shares']}")

    details_str = ", ".join(details) if details else "No Positions"

    return total_value, pl_pct, details_str

def send_discord_notification(message):
    """
    Sends a message to Discord via Webhook.
    """
    data = {
        "content": message
    }
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        if response.status_code != 204 and response.status_code != 200:
            print(f"Failed to send Discord notification: {response.status_code} {response.text}")
    except Exception as e:
        logging.error(f"Error sending Discord notification: {e}")

def parse_ai_json(response_text):
    """
    Parses JSON from Gemini response, handling markdown code blocks.
    """
    try:
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        return json.loads(clean_text.strip())
    except Exception as e:
        logging.error(f"JSON Parse Error: {e}")
        return None

@with_retry()
def get_ai_analysis(stock_metrics, context=None, history_summary="", global_sentiment="Neutral", stock_news=""):
    """
    Uses Gemini 2.5 Flash to generate a JSON decision with Self-Learning context and News Intelligence.
    Context: Optional dictionary with 'cost', 'profit_pct', 'shares' for existing holdings.
    history_summary: String summary of recent trade performance for in-context learning.
    global_sentiment: Morning macro sentiment string.
    stock_news: Specific news headlines for the stock.
    """
    logging.info(f"Requesting AI Analysis for {stock_metrics['Name']}...")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Learning Section
        learning_str = ""
        if history_summary:
            learning_str = (
                f"🔍 SELF-REFLECTION & LEARNING:\n"
                f"Review your recent trade history below. Identify patterns in your Wins and Losses.\n"
                f"Instruction: Analyze the MACD and BB% values in the history lines above.\n"
                f"- If you see repeated losses when MACD was negative or BB% < 0 (Catching Knives), avoid that setup now.\n"
                f"- If you see wins when MACD was positive (Momentum), favor that setup.\n"
                f"{history_summary}\n\n"
            )

        # Base Data
        data_str = (
            f"{learning_str}"
            f"🌍 GLOBAL MARKET SENTIMENT (Morning Briefing): {global_sentiment}\n\n"
            f"📰 LATEST STOCK NEWS:\n{stock_news}\n\n"
            f"Analyze this data for {stock_metrics['Name']} ({stock_metrics['Ticker']}):\n"
            f"Price: {stock_metrics['Price']}\n"
            f"Trend Score: {stock_metrics['Trend_Score']:.2f}\n"
            f"RSI: {stock_metrics['RSI']:.2f}\n"
            f"Foreign Net Buy: {stock_metrics['Foreign_Buy']}\n"
            f"News Sentiment Score (TextBlob): {stock_metrics['News_Sentiment_Score']:.2f}\n"
            f"Technical Indicators:\n"
            f"- MACD: {stock_metrics.get('MACD_Status', 'N/A')} (Hist: {stock_metrics.get('MACD_Hist', 0):.2f})\n"
            f"- Bollinger Bands: Price is at {stock_metrics.get('BB_Pct', 0)*100:.1f}% of the band width.\n"
            f"- Moving Averages: Price is {stock_metrics.get('MA20_Status', 'N/A')} MA20 and {stock_metrics.get('MA60_Status', 'N/A')} MA60.\n"
            f"- Volume: {stock_metrics.get('Volume_Ratio', 1.0):.2f}x vs 5-day Avg\n"
        )

        if context:
            # Holding Context
            data_str += (
                f"Current Status: HOLDING\n"
                f"Avg Cost: {context['cost']}\n"
                f"Current Profit/Loss: {context['profit_pct']:.2f}%\n"
                f"Quantity Held: {context['shares']}\n"
                f"Buy Reason: {context.get('buy_reason', 'N/A')}\n"
                f"Task: Decide whether to INCREASE position (ADD), DECREASE position (REDUCE), EXIT (CLEAR), or HOLD.\n"
            )
            json_req = (
                f"action: 'ADD', 'REDUCE', 'CLEAR', or 'HOLD'\n"
                f"percentage: float 0.1-1.0 (If ADD: % of Available Cash. If REDUCE: % of Shares to sell).\n"
            )
        else:
            # Candidate Context
            data_str += "Task: Decide whether to open a NEW position (BUY) or skip (HOLD).\n"
            json_req = (
                f"action: 'BUY' or 'HOLD'\n"
                f"percentage: float 0.01-1.00 (Percentage of available cash to invest).\n"
            )

        prompt = (
            f"You are an aggressive growth fund manager. Your goal is to achieve a 20% Monthly Return. "
            f"Prioritize High Momentum, Volatility, and Short-term Catalysts. Accept higher risks for higher rewards.\n"
            f"{data_str}"
            f"Respond strictly with a valid JSON object (no markdown) with these keys:\n"
            f"{json_req}"
            f"confidence: integer 0-100\n"
            f"reason: Brief reason in Traditional Chinese (under 30 words)."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response.text:
            return parse_ai_json(response.text)
        return None

    except Exception as e:
        logging.error(f"Gemini AI Error: {e}")
        return None

@with_retry()
def get_ai_briefing(event_type, context_data):
    """
    Generates a strategic briefing for Market Open or Close using Gemini 2.5.
    event_type: "OPEN" or "CLOSE"
    context_data: Dictionary containing market status, P/L, etc.
    """
    logging.info(f"Requesting AI Briefing for {event_type}...")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        if event_type == "OPEN":
            prompt = (
                f"You are an aggressive growth fund manager targeting 20% Monthly Return.\n"
                f"Current Market Status: {context_data.get('market_status', 'Unknown')}\n"
                f"Index Price: {context_data.get('market_price', 0)}\n"
                f"SMA60: {context_data.get('market_sma60', 0)}\n"
                f"Morning News Headlines:\n{context_data.get('macro_news', 'N/A')}\n\n"
                f"Task: The market is opening. Briefly state your strategy for today (under 100 words).\n"
                f"Synthesize the macro news and technical status.\n"
                f"- If Bullish: How will you aggressively hunt momentum?\n"
                f"- If Bearish: How will you protect capital while looking for shorts/rebounds?\n"
            )
        else: # CLOSE
            prompt = (
                f"You are an aggressive growth fund manager.\n"
                f"Market Closed.\n"
                f"Trades Executed Today: {context_data.get('trades_count', 0)}\n"
                f"Session P/L: {context_data.get('pnl_val', 0):+.0f} ({context_data.get('pnl_pct', 0):+.2f}%)\n"
                f"Current Equity: {context_data.get('equity', 0):,.0f}\n\n"
                f"Task: Briefly review today's performance and give a 1-sentence outlook/strategy for tomorrow (under 100 words)."
            )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response.text:
            return response.text.strip()
        return "AI Briefing Unavailable."

    except Exception as e:
        logging.error(f"Gemini AI Briefing Error: {e}")
        return "AI Briefing Failed."

def send_daily_briefing(event_type, context_data):
    """
    Sends the Open/Close briefing to Discord.
    """
    ai_thought = get_ai_briefing(event_type, context_data)

    if event_type == "OPEN":
        title = "☀️ **開盤通知 (Opening Bell)**"
        color = "🟢" if "Bullish" in context_data.get('market_status', '') else "🔴"
        content = (
            f"{title}\n"
            f"{color} 市場狀態: **{context_data.get('market_status')}**\n"
            f"指數: {context_data.get('market_price'):.2f} (MA60: {context_data.get('market_sma60'):.2f})\n\n"
            f"> 🧠 **Gemini 策略思路：**\n"
            f"> {ai_thought}"
        )

        # Update Global Sentiment
        global DAILY_MARKET_SENTIMENT
        DAILY_MARKET_SENTIMENT = ai_thought

    else:
        title = "🌙 **收盤報告 (Closing Bell)**"
        pnl_color = "🟢" if context_data.get('pnl_val', 0) >= 0 else "🔴"
        content = (
            f"{title}\n"
            f"📊 今日交易數: {context_data.get('trades_count')}\n"
            f"{pnl_color} 當日損益: ${context_data.get('pnl_val'):+,.0f} ({context_data.get('pnl_pct'):+.2f}%)\n"
            f"💰 總權益: ${context_data.get('equity'):,.0f}\n\n"
            f"> 📝 **Gemini 復盤與展望：**\n"
            f"> {ai_thought}"
        )

    send_discord_notification(content)

def manage_holdings(portfolio):
    """
    Step 1: Manage Holdings (ADD, REDUCE, CLEAR, HOLD).
    Returns (portfolio, list_of_trade_messages, list_of_processed_tickers).
    """
    logging.info("Step 1: Managing Holdings...")
    updated_holdings = portfolio['active_holdings'].copy()
    trade_messages = []
    processed_tickers = []
    has_sold = False

    # Calculate Total Portfolio Value for Risk Management (used in ADD)
    total_asset_value, _, _ = calculate_portfolio_value(portfolio)

    # Get History Summary
    history_summary = get_recent_performance_summary(portfolio)

    for ticker, data in portfolio['active_holdings'].items():
        processed_tickers.append(ticker)

        # Random Delay for Anti-Ban
        time.sleep(random.uniform(2.0, 4.0))

        # Handle format (migration should have ensured dict, but be safe)
        if isinstance(data, int):
            shares = data
            avg_cost = 0.0
            buy_reason = "Legacy"
        else:
            shares = data['shares']
            avg_cost = data.get('cost', 0.0)
            buy_reason = data.get('buy_reason', 'N/A')

        try:
            metrics = analyze_technicals(ticker)
            metrics['Name'] = get_stock_name_cn(ticker)

            # Real-time Price
            realtime_price = get_realtime_price(ticker)
            exec_price = realtime_price if realtime_price else metrics['Price']

            # P/L %
            profit_pct = 0.0
            if avg_cost > 0:
                profit_pct = ((exec_price - avg_cost) / avg_cost) * 100

            # AI Context
            context = {
                "cost": avg_cost,
                "profit_pct": profit_pct,
                "shares": shares,
                "buy_reason": buy_reason
            }

            # Fetch Stock News
            stock_news = fetch_stock_news(ticker)

            # AI Decision
            ai_result = get_ai_analysis(
                metrics,
                context=context,
                history_summary=history_summary,
                global_sentiment=DAILY_MARKET_SENTIMENT,
                stock_news=stock_news
            )

            action = "HOLD"
            confidence = 0
            percentage = 0.0
            reason = "N/A"

            if ai_result:
                action = ai_result.get("action", "HOLD")
                confidence = int(ai_result.get("confidence", 0))
                percentage = float(ai_result.get("percentage", 0.0))
                reason = ai_result.get("reason", "N/A")

            # --- EXECUTION LOGIC ---

            # Enforce Confidence on ADD (Aggressive but conviction required)
            if action == "ADD" and confidence <= 75:
                print(f"Skipping ADD for {ticker}: Low Confidence ({confidence}%)")
                action = "HOLD"

            # CLEAR (Sell All)
            if action == "CLEAR":
                gross_revenue = exec_price * shares
                fee = calculate_transaction_fee(gross_revenue)
                tax = calculate_tax(gross_revenue)
                net_revenue = gross_revenue - fee - tax

                portfolio['balance'] += net_revenue
                del updated_holdings[ticker]
                has_sold = True

                # Log History with Feedback Data
                log = {
                    "action": "CLEAR",
                    "ticker": ticker,
                    "price": exec_price,
                    "shares": shares,
                    "pnl_percentage": profit_pct,
                    "buy_reason": buy_reason,
                    "reason": reason, # Sell Reason
                    "indicators": {
                        "RSI": round(metrics['RSI'], 2),
                        "Trend": round(metrics['Trend_Score'], 2),
                        "MACD_Hist": round(metrics.get('MACD_Hist', 0), 4),
                        "BB_Pct": round(metrics.get('BB_Pct', 0), 2),
                        "MA_Trend": metrics.get('MA20_Status', 'N/A')
                    },
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                portfolio['history'].append(log)

                msg = f"🔴 **CLEAR**: {ticker} (All {shares} shares) @ {exec_price} | Net: ${net_revenue:,.0f} (Fee: {fee}, Tax: {tax}) | P/L: {profit_pct:.2f}% | {reason}"
                trade_messages.append(msg)
                logging.info(msg)

            # REDUCE (Sell Partial)
            elif action == "REDUCE":
                sell_ratio = max(0.1, min(percentage, 1.0)) # Clamp 10%-100%
                shares_to_sell = int(shares * sell_ratio)

                if shares_to_sell > 0:
                    gross_revenue = exec_price * shares_to_sell
                    fee = calculate_transaction_fee(gross_revenue)
                    tax = calculate_tax(gross_revenue)
                    net_revenue = gross_revenue - fee - tax

                    portfolio['balance'] += net_revenue
                    updated_holdings[ticker]['shares'] -= shares_to_sell
                    has_sold = True

                    if updated_holdings[ticker]['shares'] == 0:
                        del updated_holdings[ticker] # Fully reduced?

                    # Log
                    log = {
                        "action": "REDUCE",
                        "ticker": ticker,
                        "price": exec_price,
                        "shares": shares_to_sell,
                        "pnl_percentage": profit_pct, # Snapshot of current P/L
                        "buy_reason": buy_reason,
                        "reason": reason,
                        "indicators": {
                            "RSI": round(metrics['RSI'], 2),
                            "Trend": round(metrics['Trend_Score'], 2),
                            "MACD_Hist": round(metrics.get('MACD_Hist', 0), 4),
                            "BB_Pct": round(metrics.get('BB_Pct', 0), 2),
                            "MA_Trend": metrics.get('MA20_Status', 'N/A')
                        },
                        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    portfolio['history'].append(log)

                    msg = f"🟠 **REDUCE**: {ticker} ({shares_to_sell} shares) @ {exec_price} | Net: ${net_revenue:,.0f} | {reason}"
                    trade_messages.append(msg)
                    logging.info(msg)

            # ADD (Buy More) - Respect 40% Global Cap
            elif action == "ADD":
                alloc_ratio = max(0.01, min(percentage, 1.0)) # % of Cash
                invest_amount = portfolio['balance'] * alloc_ratio
                shares_to_buy = int(invest_amount / exec_price)

                # Check Odd Lot Minimum
                if shares_to_buy == 0 and invest_amount > exec_price:
                    shares_to_buy = 1

                # Check Global Cap
                current_position_val = shares * exec_price
                new_position_val = current_position_val + (shares_to_buy * exec_price)

                max_allowed_val = total_asset_value * MAX_POSITION_RATIO

                if new_position_val > max_allowed_val:
                    # Clamp
                    allowed_buy_val = max(0, max_allowed_val - current_position_val)
                    shares_to_buy = int(allowed_buy_val / exec_price)
                    logging.warning(f"ADD Clamped by 40% Cap for {ticker}. Reduced to {shares_to_buy} shares.")

                gross_cost = shares_to_buy * exec_price
                fee = calculate_transaction_fee(gross_cost)
                total_cost = gross_cost + fee

                if shares_to_buy > 0 and portfolio['balance'] >= total_cost:
                    portfolio['balance'] -= total_cost

                    # Update Avg Cost
                    total_shares_new = shares + shares_to_buy
                    # Cost basis includes fee
                    total_cost_old = shares * avg_cost
                    new_avg_cost = (total_cost_old + total_cost) / total_shares_new

                    updated_holdings[ticker]['shares'] = total_shares_new
                    updated_holdings[ticker]['cost'] = new_avg_cost

                    msg = f"🟢 **ADD**: {ticker} ({shares_to_buy} shares) @ {exec_price} | Cost: ${total_cost:,.0f} (Fee: {fee}) | New Avg: {new_avg_cost:.2f} | {reason}"
                    trade_messages.append(msg)
                    logging.info(msg)

            # HOLD or Unknown
            else:
                logging.info(f"AI Action for {ticker}: {action} (Holding)")

        except Exception as e:
            logging.error(f"Error managing holding {ticker}: {e}")

    portfolio['active_holdings'] = updated_holdings

    # Trigger Rescue if we sold anything (Immediate Safety Net)
    if has_sold:
        portfolio, triggered = check_capital_rescue(portfolio)

    return portfolio, trade_messages, processed_tickers

def process_buy_signals(portfolio, candidates):
    """
    Checks top candidates using AI for dynamic position sizing.
    Returns (portfolio, list_of_trade_messages).
    """
    logging.info("Step 3: Processing Buy Signals (Smart Mode)...")
    trade_messages = []

    # Calculate Total Value for 40% Cap
    total_asset_value, _, _ = calculate_portfolio_value(portfolio)

    # Get History Summary for Feedback Loop
    history_summary = get_recent_performance_summary(portfolio)

    for candidate in candidates:
        ticker = candidate['Ticker']

        # Random Delay for Anti-Ban
        time.sleep(random.uniform(2.0, 4.0))

        # Use Real-time price
        realtime_price = get_realtime_price(ticker)
        exec_price = realtime_price if realtime_price else candidate['Price']

        # Fetch Stock News
        stock_news = fetch_stock_news(ticker)

        # 1. AI Analysis
        ai_result = get_ai_analysis(
            candidate,
            history_summary=history_summary,
            global_sentiment=DAILY_MARKET_SENTIMENT,
            stock_news=stock_news
        )

        ai_action = "HOLD"
        ai_confidence = 0
        ai_percentage = 0.10
        ai_reason = "N/A"

        if ai_result:
            ai_action = ai_result.get("action", "HOLD")
            ai_confidence = int(ai_result.get("confidence", 0))
            ai_percentage = float(ai_result.get("percentage", 0.10))
            ai_reason = ai_result.get("reason", "N/A")

            # Sanitization
            if ai_percentage > 1.0:
                ai_percentage /= 100.0
            ai_percentage = max(0.01, min(ai_percentage, 1.0))

            candidate['AI_Decision'] = ai_action
            candidate['AI_Reason'] = ai_reason

        # Buy Condition: Action is BUY AND Confidence > 75 (Gatekeeper)
        if ai_action == "BUY" and ai_confidence > 75:

            # 2. Position Sizing
            invest_amount = portfolio['balance'] * ai_percentage
            shares_to_buy = int(invest_amount / exec_price)

            # Check Odd Lot Minimum
            if shares_to_buy == 0 and invest_amount > exec_price:
                shares_to_buy = 1

            # 3. Global Cap Check (40%)
            # Projected holding value = 0 (since it's new) + new buy value
            projected_value = shares_to_buy * exec_price
            max_allowed_val = total_asset_value * MAX_POSITION_RATIO

            if projected_value > max_allowed_val:
                allowed_val = max(0, max_allowed_val) # current holding is 0
                shares_to_buy = int(allowed_val / exec_price)
                print(f"BUY Clamped by 40% Cap for {ticker}. Reduced to {shares_to_buy} shares.")

            gross_cost = shares_to_buy * exec_price
            fee = calculate_transaction_fee(gross_cost)
            total_cost = gross_cost + fee

            # Final check on cash
            if shares_to_buy > 0 and portfolio['balance'] >= total_cost:
                portfolio['balance'] -= total_cost

                # New Position (Cost basis includes fee)
                # Unit cost = Total Cost / Shares
                avg_cost = total_cost / shares_to_buy

                portfolio['active_holdings'][ticker] = {
                    "shares": shares_to_buy,
                    "cost": avg_cost,
                    "buy_reason": ai_reason
                }

                reason_str = f"AI: {ai_action} | Alloc: {ai_percentage*100:.1f}% | {ai_reason}"

                # Log
                log = {
                    "action": "BUY",
                    "ticker": ticker,
                    "price": exec_price,
                    "shares": shares_to_buy,
                    "reason": reason_str,
                    "indicators": {
                        "RSI": round(candidate.get('RSI', 0), 2),
                        "Trend": round(candidate.get('Trend_Score', 0), 2),
                        "MACD_Hist": round(candidate.get('MACD_Hist', 0), 4),
                        "BB_Pct": round(candidate.get('BB_Pct', 0), 2),
                        "MA_Trend": candidate.get('MA20_Status', 'N/A')
                    },
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                portfolio['history'].append(log)

                msg = f"🟢 **BUY**: {ticker} ({shares_to_buy}股) @ {exec_price} | Cost: ${total_cost:,.0f} (Fee: {fee}) | {reason_str}"
                print(msg)
                trade_messages.append(msg)
            else:
                print(f"Skipped {ticker}: Insufficient Cash/Allocation (${invest_amount:,.0f})")
        else:
            print(f"Skipped {ticker}: AI says {ai_action} (Confidence: {ai_confidence}%)")

    return portfolio, trade_messages

def check_market_status():
    """
    Checks the Taiwan Weighted Index (^TWII) status.
    Fallback: If ^TWII fails, tries 0050.TW (Market Proxy).
    Returns (is_bullish, message, current_price, sma60).
    Strict Safety: Returns False if data fetch fails.
    """
    logging.info("Checking Market Status...")

    def _analyze_ticker(ticker_symbol):
        try:
            market = yf.Ticker(ticker_symbol)
            # Fetch 1y history to ensure enough data for SMA60
            df = market.history(period="1y")

            if df.empty:
                return None, f"{ticker_symbol} Empty"

            if 'Close' not in df.columns:
                return None, f"{ticker_symbol} No Close Col"

            if len(df) < 60:
                return None, f"{ticker_symbol} Insufficient Data ({len(df)})"

            # Calculate SMA60
            df['SMA_60'] = df.ta.sma(length=60, close='Close')

            current_close = df['Close'].iloc[-1]
            sma60 = df['SMA_60'].iloc[-1]

            if pd.isna(sma60):
                return None, f"{ticker_symbol} SMA60 NaN"

            is_bullish = current_close > sma60
            msg = "Bullish" if is_bullish else "Bearish (Below SMA60)"

            return {
                "is_bullish": is_bullish,
                "msg": msg,
                "price": current_close,
                "sma60": sma60
            }, None

        except Exception as e:
            return None, str(e)

    # 1. Try ^TWII
    result, error = _analyze_ticker("^TWII")
    if result:
        logging.info(f"Market Status (^TWII): {result['msg']}")
        return result['is_bullish'], result['msg'], result['price'], result['sma60']

    logging.warning(f"Market Check (^TWII) Failed: {error}. Trying Fallback (0050.TW)...")

    # 2. Try Fallback 0050.TW
    result, error = _analyze_ticker("0050.TW")
    if result:
        logging.info(f"Market Status (0050.TW Proxy): {result['msg']}")
        return result['is_bullish'], f"Proxy: {result['msg']}", result['price'], result['sma60']

    logging.error(f"Critical: Market Status Check Failed (Both ^TWII and 0050.TW). Error: {error}")
    return False, f"Critical Error: {error}", 0, 0

def get_stock_list():
    """
    Returns the curated Battlefield Targets list for production.
    """
    print("Loading Battlefield Targets...")
    return BATTLEFIELD_TARGETS

@with_retry()
def analyze_technicals(ticker):
    """
    Fetches history and performs technical analysis.
    Returns a dictionary of metrics.
    """
    metrics = {
        'Ticker': ticker,
        'Name': ticker, # Placeholder, will try to get name
        'Price': 0.0,
        'RSI': 0.0,
        'SMA20': 0.0,
        'SMA60': 0.0,
        'ATR': 0.0,
        'Trailing_Exit_Price': 0.0,
        'Trend_Score': 0.0,
        'Past_Month_Return': 0.0,
        'Pass_Technical': False,
        'Pass_Performance': False,
        'Error': None
    }

    try:
        # Fetch 1 year of history
        # We need slightly more than 1 year for indicators, or exactly 1y is fine for 'Past_Year_Return'
        # but 1y of data is enough for SMA60 usually if there are enough trading days.
        # Fetching '2y' to be safe for 1 full year return + indicators preamble.
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")

        if df.empty or len(df) < 60:
            metrics['Error'] = "Insufficient Data"
            return metrics

        # Calculate Technical Indicators
        # RSI 14
        df['RSI'] = df.ta.rsi(length=14)
        # SMA 20, 60
        df['SMA_20'] = df.ta.sma(length=20)
        df['SMA_60'] = df.ta.sma(length=60)

        # ATR 14 (Volatillity)
        # pandas_ta automatically uses High, Low, Close from df
        df['ATR'] = df.ta.atr(length=14)

        # Get latest values
        current_close = df['Close'].iloc[-1]
        metrics['Price'] = current_close

        # Check if indicators are NaN (not enough data)
        if pd.isna(df['RSI'].iloc[-1]) or pd.isna(df['SMA_60'].iloc[-1]) or pd.isna(df['ATR'].iloc[-1]):
             metrics['Error'] = "Indicators NaN"
             return metrics

        metrics['RSI'] = df['RSI'].iloc[-1]
        metrics['SMA20'] = df['SMA_20'].iloc[-1]
        metrics['SMA60'] = df['SMA_60'].iloc[-1]
        metrics['ATR'] = df['ATR'].iloc[-1]

        # --- Advanced Indicators ---

        # MACD (12, 26, 9)
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        # pandas_ta returns columns: MACD_12_26_9, MACDh_12_26_9 (Hist), MACDs_12_26_9 (Signal)
        metrics['MACD_Line'] = macd['MACD_12_26_9'].iloc[-1]
        metrics['MACD_Signal'] = macd['MACDs_12_26_9'].iloc[-1]
        metrics['MACD_Hist'] = macd['MACDh_12_26_9'].iloc[-1]

        if metrics['MACD_Line'] > metrics['MACD_Signal']:
            metrics['MACD_Status'] = "Bullish"
        else:
            metrics['MACD_Status'] = "Bearish"

        # Bollinger Bands (20, 2)
        bbands = df.ta.bbands(length=20, std=2)
        # Columns: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBP_20_2.0 (%B)
        metrics['BB_Pct'] = bbands['BBP_20_2.0'].iloc[-1]

        # Volume Spike (Current vs SMA5)
        # Check if Volume column exists and handle potential 0
        if 'Volume' in df.columns:
            vol_sma5 = df['Volume'].rolling(window=5).mean().iloc[-1]
            current_vol = df['Volume'].iloc[-1]
            if vol_sma5 > 0:
                metrics['Volume_Ratio'] = current_vol / vol_sma5
            else:
                metrics['Volume_Ratio'] = 1.0
        else:
            metrics['Volume_Ratio'] = 1.0

        metrics['MA20_Status'] = "Above" if current_close > metrics['SMA20'] else "Below"
        metrics['MA60_Status'] = "Above" if current_close > metrics['SMA60'] else "Below"

        # Trailing Exit: Highest High (20d) - 3 * ATR
        # We need rolling max of High for 20 days
        highest_high_20 = df['High'].tail(20).max()
        metrics['Trailing_Exit_Price'] = highest_high_20 - (3 * metrics['ATR'])

        # Trend Score: ((Close - SMA60) / SMA60) * 100
        metrics['Trend_Score'] = ((current_close - metrics['SMA60']) / metrics['SMA60']) * 100

        # Past Month Return (Aggressive Growth)
        # We need the price from roughly 1 month ago (21 trading days)
        if len(df) > 21:
            price_1m_ago = df['Close'].iloc[-21]
            metrics['Past_Month_Return'] = (current_close - price_1m_ago) / price_1m_ago
        else:
            # If less than a month, use the earliest available
            price_start = df['Close'].iloc[0]
            metrics['Past_Month_Return'] = (current_close - price_start) / price_start

        # Technical Filter: Price > SMA20 > SMA60 (Bullish alignment) AND RSI < 85
        if (current_close > metrics['SMA20'] > metrics['SMA60']) and (metrics['RSI'] < 85):
            metrics['Pass_Technical'] = True

        # Performance Filter: Past_Month_Return >= TARGET_MONTHLY_RETURN
        if metrics['Past_Month_Return'] >= TARGET_MONTHLY_RETURN:
            metrics['Pass_Performance'] = True

        return metrics

    except Exception as e:
        metrics['Error'] = str(e)
        return metrics

@with_retry()
def get_chips_data(stock_id):
    """
    Fetches Foreign and Investment Trust Net Buy data using FinMind.
    Returns (foreign_buy, trust_buy) tuple.
    """
    try:
        # stock_id needs to be just the code (e.g., "2330"), not "2330.TW"
        clean_id = stock_id.split('.')[0]

        dl = DataLoader()
        # Fetching Institutional Investors Buy Sell
        # Table: TaiwanStockInstitutionalInvestorsBuySell

        # We try to fetch the last few days to find the latest valid data
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime('%Y-%m-%d')

        df = dl.taiwan_stock_institutional_investors_buy_sell(
            stock_id=clean_id,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            return 0, 0

        # Get the latest date's data
        latest_date = df['date'].max()
        latest_data = df[df['date'] == latest_date]

        # 'name' column contains "Foreign_Investor", "Investment_Trust", etc.
        # 'buy' and 'sell' columns (shares). Net = buy - sell.
        # FinMind structure usually has 'name', 'buy', 'sell'

        # Note: FinMind names can be in Chinese or English depending on version.
        # Common names: 'Foreign_Investor' (外資), 'Investment_Trust' (投信)
        # Check unique names to be safe if possible, but assuming standard names or Chinese

        foreign_buy = 0
        trust_buy = 0

        for index, row in latest_data.iterrows():
            name = row['name']
            net_buy = row['buy'] - row['sell']

            # Map names
            if 'Foreign' in name or '外資' in name:
                foreign_buy += net_buy
            elif 'Trust' in name or '投信' in name:
                trust_buy += net_buy

        return foreign_buy, trust_buy

    except Exception as e:
        # print(f"FinMind Error for {stock_id}: {e}") # Optional debug
        return 0, 0

@with_retry()
def get_news_sentiment(ticker):
    """
    Fetches top 3 news headlines from yfinance and calculates average sentiment.
    Returns (average_score, latest_headline).
    """
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news

        if not news_list:
            return 0.0, "No News Found"

        scores = []
        headlines = []

        # Analyze up to 3 latest news items
        for item in news_list[:3]:
            title = item.get('title', '')
            if title:
                headlines.append(title)
                blob = TextBlob(title)
                scores.append(blob.sentiment.polarity)

        if not scores:
            return 0.0, "No News Found"

        avg_score = sum(scores) / len(scores)
        latest_headline = headlines[0] if headlines else "No News"

        return avg_score, latest_headline

    except Exception as e:
        # print(f"News Error for {ticker}: {e}") # Optional debug
        return 0.0, "Error Fetching News"

def main():
    print("Starting Stock Analysis Tool...")

    # Initialization for Daily P/L Tracking
    start_portfolio = load_portfolio()
    session_start_equity, _, _ = calculate_portfolio_value(start_portfolio)
    trades_executed_count = 0
    has_sent_open_notification = False

    if INTRADAY_MODE:
        send_discord_notification("🟢 **Bot Started**: 盤中監控模式已啟動 (每 5 分鐘掃描一次)\n🚀 目標策略：月報酬 20% (高風險高報酬模式)")

    # Main Intraday Loop
    while True:
        try:
            # Time Check (Taiwan Time: UTC+8)
            tz_tw = datetime.timezone(datetime.timedelta(hours=8))
            now = datetime.datetime.now(tz_tw)

            # Schedules
            morning_briefing_time = now.replace(hour=8, minute=30, second=0, microsecond=0)
            market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
            market_close = now.replace(hour=13, minute=30, second=0, microsecond=0)

            if INTRADAY_MODE:
                print(f"Current Time: {now.strftime('%H:%M:%S')}")

                # 08:30 - 09:00: MORNING BRIEFING
                # Run this only once per day. Logic: If it's past 08:30 and before 09:00 and we haven't sent it.
                # However, to keep it simple in this loop, we can trigger it if current time is within window
                # and relying on a 'has_sent_open_notification' flag which we already have, but that was for 09:00.

                # We reuse 'has_sent_open_notification' for the 08:30 briefing instead of 09:00 open.
                # Reset logic if needed, but for now we assume bot restarts or runs continuously.
                if not has_sent_open_notification and (morning_briefing_time <= now < market_close):
                    print("Generating Morning Briefing...")

                    # Fetch Macro News
                    macro_news = fetch_macro_news()

                    # Check Market Technicals (Previous Close)
                    is_bullish, market_msg, market_price, market_sma60 = check_market_status()

                    open_context = {
                        "market_status": market_msg,
                        "market_price": market_price,
                        "market_sma60": market_sma60,
                        "macro_news": macro_news
                    }

                    send_daily_briefing("OPEN", open_context)
                    has_sent_open_notification = True

                # PRE-MARKET WAIT
                if now < market_open:
                    print("Pre-market. Waiting...")
                    time.sleep(60)
                    continue

                # MARKET CLOSED -> EXIT
                elif now > market_close:
                    print("Market Closed. Preparing Closing Report...")

                    # Close Briefing
                    end_portfolio = load_portfolio()

                    # 1. Run Capital Rescue (Safety Net)
                    end_portfolio, _ = check_capital_rescue(end_portfolio)

                    # 2. Run AI Profit Harvesting (The Manager)
                    end_portfolio = check_profit_harvest_ai(end_portfolio)

                    save_portfolio(end_portfolio)

                    current_equity, _, _ = calculate_portfolio_value(end_portfolio)
                    pnl_val = current_equity - session_start_equity
                    pnl_pct = 0
                    if session_start_equity > 0:
                        pnl_pct = (pnl_val / session_start_equity) * 100

                    context = {
                        "trades_count": trades_executed_count,
                        "pnl_val": pnl_val,
                        "pnl_pct": pnl_pct,
                        "equity": current_equity
                    }
                    send_daily_briefing("CLOSE", context)

                    print("Exiting Bot.")
                    break

            # --- START CYCLE ---

            # Load Portfolio
            portfolio = load_portfolio()

            all_messages = []

            # Phase 1: Manage Holdings (Sell/Add/Reduce)
            # This logic runs REGARDLESS of market status (Step 1)
            portfolio, holding_msgs, processed_tickers = manage_holdings(portfolio)
            all_messages.extend(holding_msgs)
            if holding_msgs:
                trades_executed_count += len(holding_msgs)

            # Phase 2: Market Regime Filter
            is_bullish, market_msg, market_price, market_sma60 = check_market_status()

            # OPENING BELL (Redundant if Morning Briefing already sent, but keeping logic safe)
            # Removed the duplicate open notification block since it's handled above at 08:30

            if not is_bullish:
                # Market is Bearish or Error
                print(f"⚠️ Market Bearish/Error: {market_msg}. Skipping Buy Scan.")

                # If we had holding messages, we should notify
                if holding_msgs:
                    halt_msg = (
                        f"**⚠️ 系統暫停 (System Halted)**\n"
                        f"市場狀態: {market_msg}\n"
                        f"**Transactions (Holdings):**\n" + "\n".join(holding_msgs)
                    )
                    send_discord_notification(halt_msg)

                # Save & Sleep
                save_portfolio(portfolio)

                if INTRADAY_MODE:
                    time.sleep(300)
                    continue
                else:
                    return

            print(f"Market Status: Bullish. Proceeding to Scan...")

            # Phase 3: Scan New Opportunities
            stocks = get_stock_list()

            all_results = []
            candidates = []

            # Step 2: Scan (Use Battlefield Targets)
            print("Step 2: Technical & Performance Scan (Candidates)")

            # Shuffle for fairness in latency
            scan_list = list(stocks)
            random.shuffle(scan_list)

            for ticker in tqdm(scan_list):
                # Check if processed in Step 1 (Holdings)
                if ticker in processed_tickers:
                    continue

                # Random Delay for Anti-Ban
                time.sleep(random.uniform(2.0, 4.0))

                # Step A: Analyze Technicals
                metrics = analyze_technicals(ticker)

                # Populate Name (Chinese if available)
                metrics['Name'] = get_stock_name_cn(ticker)

                all_results.append(metrics)

                # Step B: Check Filters
                if metrics['Pass_Technical'] and metrics['Pass_Performance']:
                    # Passed Initial Filters -> Deep Dive

                    # Step C: Chips Analysis
                    f_buy, t_buy = get_chips_data(ticker)
                    metrics['Foreign_Buy'] = f_buy
                    metrics['Trust_Buy'] = t_buy

                    # Step D: News Sentiment
                    news_score, headline = get_news_sentiment(ticker)
                    metrics['News_Sentiment_Score'] = news_score
                    metrics['Latest_Headline'] = headline

                    # Step E: News Safety Filter
                    if news_score < -0.2:
                        metrics['Pass_Safety'] = False
                        metrics['Rejection_Reason'] = "Negative News Sentiment"
                    else:
                        metrics['Pass_Safety'] = True

                        # Bonus for Positive News
                        if news_score > 0.1:
                            metrics['Trend_Score'] += 10

                        candidates.append(metrics)
                else:
                    # Did not pass initial filters
                    metrics['Foreign_Buy'] = 0
                    metrics['Trust_Buy'] = 0
                    metrics['News_Sentiment_Score'] = 0.0
                    metrics['Latest_Headline'] = ""
                    metrics['Pass_Safety'] = False

            # Save All Results (Technical Scan)
            # In loop mode, maybe we don't want to write CSV every 5 mins?
            # Or overwrite is fine.
            df_all = pd.DataFrame(all_results)
            df_all.to_csv('all_technical_scan.csv', index=False)

            # Process Candidates (Portfolio Allocation)
            print(f"Found {len(candidates)} candidates passing all filters.")

            if candidates:
                # Sort by Trend Score (Descending)
                df_candidates = pd.DataFrame(candidates)
                df_candidates = df_candidates.sort_values(by='Trend_Score', ascending=False)

                # Pick Top 5
                top_picks = df_candidates.head(5).copy()

                # Phase 3: Buy Signals (Paper Trading)
                top_candidates_list = top_picks.to_dict('records')
                portfolio, buy_msgs = process_buy_signals(portfolio, top_candidates_list)
                all_messages.extend(buy_msgs)
                if buy_msgs:
                    trades_executed_count += len(buy_msgs)

                # Save Portfolio
                save_portfolio(portfolio)

                # Generate Recommendations CSV (for record)
                # Recalculate static sizing just for the CSV output (display only)
                # Since we use dynamic sizing now, this part of the CSV is less relevant but kept for structure
                # We can remove the old static logic block or keep it for the CSV display.
                # Keeping the CSV export simple.
                top_picks.to_csv('final_buy_recommendations.csv', index=False)

            # Notification Logic (Quiet Mode)
            if all_messages:
                # Calculate Value for notification
                total_val, pl_pct, details = calculate_portfolio_value(portfolio)

                msg_body = "\n".join(all_messages)
                msg = (
                    f"**【AI 交易通知】**\n"
                    f"{msg_body}\n\n"
                    f"**📊 模擬帳戶**\n"
                    f"淨值: ${total_val:,.0f} ({pl_pct:+.2f}%)\n"
                    f"現金: ${portfolio['balance']:,.0f}"
                )
                send_discord_notification(msg)
            else:
                print("No trades executed this cycle.")

            if not INTRADAY_MODE:
                break

            print("Cycle complete. Sleeping 5 minutes...")
            time.sleep(300)

        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            print(f"Unexpected Error in Main Loop: {e}")
            if INTRADAY_MODE:
                time.sleep(60) # Short sleep on error
            else:
                break

if __name__ == "__main__":
    main()
