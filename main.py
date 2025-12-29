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

# Constants
INITIAL_CAPITAL = 1000000
TARGET_ANNUAL_RETURN = 0.15
TEST_MODE = False

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

        # Get latest values
        current_close = df['Close'].iloc[-1]
        metrics['Price'] = current_close

        # Check if indicators are NaN (not enough data)
        if pd.isna(df['RSI'].iloc[-1]) or pd.isna(df['SMA_60'].iloc[-1]):
             metrics['Error'] = "Indicators NaN"
             return metrics

        metrics['RSI'] = df['RSI'].iloc[-1]
        metrics['SMA20'] = df['SMA_20'].iloc[-1]
        metrics['SMA60'] = df['SMA_60'].iloc[-1]

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

    # 1. Get Stock List
    stocks = get_stock_list()

    all_results = []
    candidates = []

    print("Step 1: Technical & Performance Scan")
    for ticker in tqdm(stocks):
        # Step A: Analyze Technicals
        metrics = analyze_technicals(ticker)
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

    # Allocation
    num_picks = len(top_picks)
    budget_per_stock = INITIAL_CAPITAL / num_picks

    top_picks['Allocated_Budget'] = budget_per_stock
    # Calculate shares: floor division
    top_picks['Suggested_Shares'] = (top_picks['Allocated_Budget'] / top_picks['Price']).astype(int)

    # Save Final Recommendations
    print("Saving final buy recommendations...")
    top_picks.to_csv('final_buy_recommendations.csv', index=False)

    # Print Console Table
    print("\n=== TOP BUY RECOMMENDATIONS ===")
    cols_to_show = ['Ticker', 'Price', 'Trend_Score', 'Past_Year_Return', 'Foreign_Buy', 'News_Sentiment_Score', 'Suggested_Shares']
    print(top_picks[cols_to_show].to_string(index=False))
    print("===============================")

if __name__ == "__main__":
    main()
