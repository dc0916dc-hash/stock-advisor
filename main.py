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

# Constants
INITIAL_CAPITAL = 1000000
TARGET_ANNUAL_RETURN = 0.15
TEST_MODE = False
LINE_TOKEN = "YOUR_TOKEN_HERE"

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

def send_line_notification(message):
    """
    Sends a message to LINE Notify.
    """
    url = 'https://notify-api.line.me/api/notify'
    headers = {
        'Authorization': f'Bearer {LINE_TOKEN}'
    }
    data = {
        'message': message
    }
    try:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code != 200:
            print(f"Failed to send LINE notification: {response.status_code}")
    except Exception as e:
        print(f"Error sending LINE notification: {e}")

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
    print(f"Configuration: Capital={INITIAL_CAPITAL}, Target Return={TARGET_ANNUAL_RETURN}, TEST_MODE={TEST_MODE}")

    # 0. Market Regime Filter
    is_bullish, market_msg, market_price, market_sma60 = check_market_status()

    if not is_bullish:
        # Market is Bearish or Error
        print(f"⚠️ SYSTEM HALTED: {market_msg}")
        halt_msg = f"\n⚠️ 系統暫停 (System Halted): 無法偵測大盤趨勢或大盤位於空頭 (^TWII < 60MA)。為保護資金，今日暫停選股。\nStatus: {market_msg}\nPrice: {market_price:.2f}, SMA60: {market_sma60:.2f}"
        send_line_notification(halt_msg)
        return

    print(f"Market Status: Bullish (Price: {market_price:.2f} > SMA60: {market_sma60:.2f}). Proceeding...")

    # 1. Get Stock List
    stocks = get_stock_list()

    all_results = []
    candidates = []

    print("Step 1: Technical & Performance Scan")
    for ticker in tqdm(stocks):
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
    print("Saving all technical scan results...")
    df_all = pd.DataFrame(all_results)
    df_all.to_csv('all_technical_scan.csv', index=False)

    # Process Candidates (Portfolio Allocation)
    print(f"Found {len(candidates)} candidates passing all filters.")

    if not candidates:
        print("No stocks passed all filters. Exiting.")
        return

    # Sort by Trend Score (Descending)
    df_candidates = pd.DataFrame(candidates)
    df_candidates = df_candidates.sort_values(by='Trend_Score', ascending=False)

    # Pick Top 5
    top_picks = df_candidates.head(5).copy()

    # Allocation Logic (ATR + Cap)
    # Risk per stock = 1% of Capital
    # Stop Loss Distance = 2 * ATR
    # Risk_Based_Shares = Risk_Amount / Stop_Loss_Distance
    # Max_Cap_Shares = (Capital * 20%) / Price

    risk_per_trade = INITIAL_CAPITAL * 0.01
    max_position_value = INITIAL_CAPITAL * 0.20

    suggested_shares_list = []
    allocated_budget_list = [] # Risk_Amount_NTD (Actual Position Value)
    stop_loss_list = []

    for index, row in top_picks.iterrows():
        price = row['Price']
        atr = row['ATR']

        # Stop Loss Price
        stop_loss_price = price - (2 * atr)
        stop_loss_list.append(round(stop_loss_price, 2))

        # Sizing
        stop_loss_dist = 2 * atr
        if stop_loss_dist > 0:
             risk_shares = risk_per_trade / stop_loss_dist
        else:
             risk_shares = 0

        max_cap_shares = max_position_value / price

        final_shares = int(min(risk_shares, max_cap_shares))
        suggested_shares_list.append(final_shares)

        allocated_budget_list.append(final_shares * price)

    top_picks['Suggested_Shares'] = suggested_shares_list
    top_picks['Risk_Amount_NTD'] = allocated_budget_list
    top_picks['Stop_Loss_Price'] = stop_loss_list

    # Save Final Recommendations
    print("Saving final buy recommendations...")
    # Add ATR to output if not already there (it is in candidates, so it's in top_picks)
    top_picks.to_csv('final_buy_recommendations.csv', index=False)

    # Print Console Table
    print("\n=== TOP BUY RECOMMENDATIONS ===")
    cols_to_show = ['Ticker', 'Name', 'Price', 'Trend_Score', 'ATR', 'Stop_Loss_Price', 'Trailing_Exit_Price', 'Suggested_Shares', 'Risk_Amount_NTD']
    print(top_picks[cols_to_show].to_string(index=False))
    print("===============================")

    # LINE Notification
    if not top_picks.empty:
        top_1_name = top_picks.iloc[0]['Name']
        top_1_score = round(top_picks.iloc[0]['Trend_Score'], 2)

        # Get list of top 3 tickers
        top_3_tickers = top_picks.head(3)['Name'].tolist() # Using Name is better
        top_3_str = ", ".join([str(x) for x in top_3_tickers])

        msg = (
            f"\n【AI 投資日報】 分析完成！\n"
            f"市場狀態: 多頭 (Bullish)\n"
            f"選出強勢股：[{top_3_str}]\n"
            f"最高分：{top_1_name} (Score: {top_1_score})\n"
            f"請查看雲端報表。"
        )
        print("Sending LINE notification...")
        send_line_notification(msg)

if __name__ == "__main__":
    main()
