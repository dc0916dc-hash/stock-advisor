import yfinance as yf
import pandas as pd
import pandas_ta as ta
import twstock
from tqdm import tqdm
from FinMind.data import DataLoader
from textblob import TextBlob
import time
import sys
import datetime
import requests
import os
import json
from google import genai

# Constants
INITIAL_CAPITAL = 1000000
TARGET_ANNUAL_RETURN = 0.15
TEST_MODE = True
INTRADAY_MODE = True
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PORTFOLIO_FILE = "portfolio.json"

# Validate Keys
if not DISCORD_WEBHOOK_URL:
    raise ValueError("Missing Environment Variable: DISCORD_WEBHOOK_URL. Please set this in GitHub Secrets.")
if not GEMINI_API_KEY:
    raise ValueError("Missing Environment Variable: GEMINI_API_KEY. Please set this in GitHub Secrets.")

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
        print(f"Error fetching realtime price for {ticker}: {e}")
        return None

def load_portfolio():
    """
    Loads portfolio from JSON file or creates a default one if not exists.
    """
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Error decoding portfolio.json, creating new.")

    # Default Portfolio
    return {
        "balance": 100000,  # 100,000 TWD Initial Capital
        "holdings": {},     # {"Ticker": Shares}
        "history": []       # List of transaction logs
    }

def save_portfolio(portfolio):
    """
    Saves portfolio data to JSON file.
    """
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=4)

def calculate_portfolio_value(portfolio):
    """
    Calculates total portfolio value (Cash + Market Value of Holdings).
    Returns (total_value, profit_loss_pct, details_str).
    """
    total_value = portfolio['balance']
    holdings_value = 0
    details = []

    for ticker, shares in portfolio['holdings'].items():
        try:
            # Fetch current price
            # Note: In a real loop, we might have this data already, but for safety fetching fresh
            stock = yf.Ticker(ticker)
            # Use 'fast' fetch if possible, history(period='1d')
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                value = current_price * shares
                holdings_value += value
                details.append(f"{ticker}: {shares}股 (${value:,.0f})")
            else:
                # Fallback if no data (delisted?), assume cost basis or last known?
                # For safety, just keep as 0 but warn
                details.append(f"{ticker}: {shares}股 (No Data)")
        except Exception:
             details.append(f"{ticker}: {shares}股 (Error)")

    total_value += holdings_value

    initial_capital = 100000
    pl_pct = ((total_value - initial_capital) / initial_capital) * 100

    return total_value, pl_pct, ", ".join(details)

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
        print(f"Error sending Discord notification: {e}")

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
        print(f"JSON Parse Error: {e}")
        return None

def get_ai_analysis(stock_metrics, purpose="BUY"):
    """
    Uses Gemini 2.5 Flash to generate a JSON decision.
    Purpose: "BUY" (for candidates) or "SELL" (for holdings).
    """
    print(f"Requesting AI Decision for {stock_metrics['Name']} ({purpose})...")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = (
            f"You are a professional fund manager trading Taiwan stocks. "
            f"Analyze this data for {stock_metrics['Name']} ({stock_metrics['Ticker']}) to make a {purpose} decision:\n"
            f"Price: {stock_metrics['Price']}\n"
            f"Trend Score: {stock_metrics['Trend_Score']:.2f}\n"
            f"RSI: {stock_metrics['RSI']:.2f}\n"
            f"Foreign Net Buy: {stock_metrics['Foreign_Buy']}\n"
            f"News Sentiment: {stock_metrics['News_Sentiment_Score']:.2f}\n"
            f"Respond strictly with a valid JSON object (no markdown) with these keys:\n"
            f"decision: 'BUY', 'SELL', or 'HOLD'\n"
            f"confidence: integer 0-100\n"
            f"reason: Brief reason in Traditional Chinese (under 30 words).\n"
            f"allocation_percent: float 0.01-1.00 (Percentage of available cash to invest)."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response.text:
            return parse_ai_json(response.text)
        return None

    except Exception as e:
        print(f"Gemini AI Error: {e}")
        return None

def process_sell_signals(portfolio):
    """
    Checks existing holdings for sell signals (Trend_Score < 0).
    Returns (portfolio, list_of_trade_messages).
    """
    print("Processing Sell Signals for Holdings...")
    updated_holdings = portfolio['holdings'].copy()
    trade_messages = []

    for ticker, shares in portfolio['holdings'].items():
        try:
            # We need to analyze the stock again to get the Trend_Score and AI check
            metrics = analyze_technicals(ticker)
            metrics['Name'] = get_stock_name_cn(ticker) # Fetch name for AI

            # Use Real-time price for execution
            realtime_price = get_realtime_price(ticker)
            exec_price = realtime_price if realtime_price else metrics['Price']

            # 1. Technical Sell Check
            tech_sell = metrics['Trend_Score'] < -5 # Hard Stop if Trend crashes

            # 2. AI Sell Check
            ai_sell = False
            ai_reason = "Technical Stop"

            ai_result = get_ai_analysis(metrics, purpose="SELL")
            if ai_result:
                decision = ai_result.get("decision", "HOLD")
                if decision == "SELL":
                    ai_sell = True
                    ai_reason = f"AI: {ai_result.get('reason', 'Sell')}"

            # Execute Sell if EITHER condition is true
            if tech_sell or ai_sell:
                revenue = exec_price * shares
                portfolio['balance'] += revenue
                del updated_holdings[ticker]

                reason_str = ai_reason if ai_sell else "Trend Score < -5"

                # Log
                log = {
                    "action": "SELL",
                    "ticker": ticker,
                    "price": exec_price,
                    "shares": shares,
                    "reason": reason_str,
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                portfolio['history'].append(log)

                msg = f"🔴 **SELL**: {ticker} ({shares}股) @ {exec_price} | {reason_str}"
                print(msg)
                trade_messages.append(msg)

        except Exception as e:
            print(f"Error checking sell signal for {ticker}: {e}")

    portfolio['holdings'] = updated_holdings
    return portfolio, trade_messages

def process_buy_signals(portfolio, candidates):
    """
    Checks top candidates using AI for dynamic position sizing.
    Returns (portfolio, list_of_trade_messages).
    """
    print("Processing Buy Signals (Smart Mode)...")
    trade_messages = []

    for candidate in candidates:
        ticker = candidate['Ticker']

        # Duplicate Prevention: Do not buy if already owned
        if ticker in portfolio['holdings']:
            print(f"Skipping {ticker}: Already in portfolio.")
            continue

        # Use Real-time price for execution if possible
        realtime_price = get_realtime_price(ticker)
        exec_price = realtime_price if realtime_price else candidate['Price']

        # 1. AI Analysis
        ai_result = get_ai_analysis(candidate, purpose="BUY")

        ai_decision = "HOLD"
        ai_confidence = 0
        ai_allocation = 0.10 # Default 10%

        if ai_result:
            ai_decision = ai_result.get("decision", "HOLD")
            ai_confidence = int(ai_result.get("confidence", 0))
            raw_allocation = ai_result.get("allocation_percent", 0.10)

            # Sanitization & Clamping
            try:
                ai_allocation = float(raw_allocation)
                if ai_allocation > 1.0:
                    ai_allocation /= 100.0
                ai_allocation = max(0.01, min(ai_allocation, 1.0))
            except:
                ai_allocation = 0.10

            # Store for notification
            candidate['AI_Decision'] = ai_decision
            candidate['AI_Confidence'] = ai_confidence

        # Buy Condition: AI says BUY and Confidence > 75
        if ai_decision == "BUY" and ai_confidence > 75:

            # 2. Dynamic Position Sizing (Based on Remaining Cash)
            current_cash = portfolio['balance']
            invest_amount = current_cash * ai_allocation

            # Calculate shares
            shares_to_buy = int(invest_amount / exec_price)

            # Round down to nearest 10 if possible, else 1
            if shares_to_buy >= 10:
                shares_to_buy = (shares_to_buy // 10) * 10

            cost = shares_to_buy * exec_price

            # Final check on cash
            if shares_to_buy > 0 and portfolio['balance'] >= cost:
                portfolio['balance'] -= cost

                if ticker in portfolio['holdings']:
                    portfolio['holdings'][ticker] += shares_to_buy
                else:
                    portfolio['holdings'][ticker] = shares_to_buy

                reason_str = f"AI: {ai_decision} ({ai_confidence}%) | Alloc: {ai_allocation*100:.1f}%"

                # Log
                log = {
                    "action": "BUY",
                    "ticker": ticker,
                    "price": exec_price,
                    "shares": shares_to_buy,
                    "reason": reason_str,
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                portfolio['history'].append(log)

                msg = f"🟢 **BUY**: {ticker} ({shares_to_buy}股) @ {exec_price} | {reason_str}"
                print(msg)
                trade_messages.append(msg)
            else:
                print(f"Skipped {ticker}: Insufficient Cash for AI allocation (${invest_amount:,.0f})")
        else:
            print(f"Skipped {ticker}: AI says {ai_decision} ({ai_confidence}%)")

    return portfolio, trade_messages

def check_market_status():
    """
    Checks the Taiwan Weighted Index (^TWII) status.
    Returns (is_bullish, message, current_price, sma60).
    Strict Safety: Returns False if data fetch fails.
    """
    print("Checking Market Status (^TWII)...")
    try:
        market = yf.Ticker("^TWII")
        # Fetch 1y history to ensure enough data for SMA60
        df = market.history(period="1y")

        if df.empty or len(df) < 60:
            return False, "Insufficient Market Data", 0, 0

        # Calculate SMA60
        df['SMA_60'] = df.ta.sma(length=60, close='Close')

        current_close = df['Close'].iloc[-1]
        sma60 = df['SMA_60'].iloc[-1]

        if pd.isna(sma60):
            return False, "Market SMA60 NaN", current_close, 0

        if current_close > sma60:
            return True, "Bullish", current_close, sma60
        else:
            return False, "Bearish (Below SMA60)", current_close, sma60

    except Exception as e:
        print(f"Market Check Error: {e}")
        return False, f"Error: {str(e)}", 0, 0

def get_stock_list():
    """
    Generates a list of Taiwan stock tickers with appropriate suffixes (.TW or .TWO).
    """
    print("Generating stock list...")
    tickers = []

    # Iterate through all codes in twstock
    for code, info in twstock.codes.items():
        # specific to '股票' (stocks) and exclude warrants/others if possible
        # twstock type: '股票', 'ETF', etc.
        if info.type == '股票':
            if info.market == '上市':
                tickers.append(f"{code}.TW")
            elif info.market == '上櫃':
                tickers.append(f"{code}.TWO")

    print(f"Total stocks found: {len(tickers)}")

    if TEST_MODE:
        print(f"TEST_MODE is ON. Slicing to top 50 stocks.")
        return tickers[:50]

    return tickers

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
        'Past_Year_Return': 0.0,
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

        # Trailing Exit: Highest High (20d) - 3 * ATR
        # We need rolling max of High for 20 days
        highest_high_20 = df['High'].tail(20).max()
        metrics['Trailing_Exit_Price'] = highest_high_20 - (3 * metrics['ATR'])

        # Trend Score: ((Close - SMA60) / SMA60) * 100
        metrics['Trend_Score'] = ((current_close - metrics['SMA60']) / metrics['SMA60']) * 100

        # Past Year Return
        # We need the price from roughly 1 year ago (252 trading days)
        if len(df) > 252:
            price_1y_ago = df['Close'].iloc[-252]
            metrics['Past_Year_Return'] = (current_close - price_1y_ago) / price_1y_ago
        else:
            # If less than a year, use the earliest available
            price_start = df['Close'].iloc[0]
            metrics['Past_Year_Return'] = (current_close - price_start) / price_start

        # Technical Filter: Price > SMA20 > SMA60 (Bullish alignment) AND RSI < 85
        if (current_close > metrics['SMA20'] > metrics['SMA60']) and (metrics['RSI'] < 85):
            metrics['Pass_Technical'] = True

        # Performance Filter: Past_Year_Return >= TARGET_ANNUAL_RETURN
        if metrics['Past_Year_Return'] >= TARGET_ANNUAL_RETURN:
            metrics['Pass_Performance'] = True

        return metrics

    except Exception as e:
        metrics['Error'] = str(e)
        return metrics

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
    if INTRADAY_MODE:
        send_discord_notification("🟢 **Bot Started**: 盤中監控模式已啟動 (每 5 分鐘掃描一次)")

    # Main Intraday Loop
    while True:
        try:
            # Time Check (Taiwan Time: UTC+8)
            tz_tw = datetime.timezone(datetime.timedelta(hours=8))
            now = datetime.datetime.now(tz_tw)

            # Simple Market Hours Check (09:00 - 13:30)
            # This simplification assumes script is run on trading days.
            market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
            market_close = now.replace(hour=13, minute=30, second=0, microsecond=0)

            if INTRADAY_MODE:
                print(f"Current Time: {now.strftime('%H:%M:%S')}")
                if now < market_open:
                    print("Pre-market. Waiting...")
                    time.sleep(60)
                    continue
                elif now > market_close:
                    print("Market Closed. Saving and Exiting.")
                    # Load and Save one last time to be safe? (Not really needed if we saved in loop)
                    break

            # --- START CYCLE ---

            # Load Portfolio
            portfolio = load_portfolio()

            all_messages = []

            # Phase 1: Sell Check (Run regardless of market status)
            portfolio, sell_msgs = process_sell_signals(portfolio)
            all_messages.extend(sell_msgs)

            # 0. Market Regime Filter
            is_bullish, market_msg, market_price, market_sma60 = check_market_status()

            if not is_bullish:
                # Market is Bearish or Error
                print(f"⚠️ Market Bearish/Error: {market_msg}. Skipping Buy Scan.")

                # If we had sell messages, we should notify
                if sell_msgs:
                    halt_msg = (
                        f"**⚠️ 系統暫停 (System Halted)**\n"
                        f"市場狀態: {market_msg}\n"
                        f"**Transactions:**\n" + "\n".join(sell_msgs)
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

            # 1. Get Stock List
            stocks = get_stock_list()

            all_results = []
            candidates = []

            print("Step 1: Technical & Performance Scan")
            # If Intraday, maybe we don't need tqdm as it clogs logs? Keeping it for now.
            for ticker in tqdm(stocks):
                # Optimization: Check if we already hold it BEFORE scanning?
                # Actually user requirement said check in buy logic, but checking here saves API calls.
                # Requirement: "Duplicate Prevention: inside the loop, before analyzing a stock, check: if stock_id in portfolio["holdings"]: continue."
                if ticker in portfolio['holdings']:
                    continue

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
