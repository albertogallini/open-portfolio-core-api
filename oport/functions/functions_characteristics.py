import yfinance as yf
import pandas as pd
import datetime
import numpy as np

from oport.fields import *


TYPE_CASH = "cash"
TYPE_RATIO = "ratio"
TYPE_AMOUNT = "amount"

"""
#Market Cap
#P/E
#P/B
#P/FCF
#EPS
#RAE
#ROE
#ROI
"""

# Cache dictionaries
financial_cache = {}
cashflow_cache = {}
closing_prices_cache = {}
dividends_cache = {}


def get_quarter(date):
    quarter = (date.month - 1) // 3 + 1
    return f"{date.year}-Q{quarter}"


def get_char_indicator(ticker_symbol: str, target_date: datetime.date, char_fields: list) -> dict:
    """
    This function computes the characteristcs for a given ticker as-of the target date.
    All the indicators are
     - computed using quarterly data,
     - aggreagted over time (TTM where applies) and
     - returned in a dict where the indicator names are the keys and a tuple (value,type)
       is the referred values.

    This function uses a global cache to make sure to hit the data-source only when is needed.
    """

    global financial_cache, cashflow_cache, closing_prices_cache, dividends_cache

    def get_closing_price(ticker, date) -> pd.Series:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(start=date - pd.Timedelta(days=365), end=date)
        history.index = pd.to_datetime(history.index).date
        return history["Close"], history["Dividends"]

    # Create a key for caching
    cache_key = (ticker_symbol, get_quarter(target_date))
    result = dict()
    
    # Create a ticker object (needed for estimates even if financials are cached)
    ticker = yf.Ticker(ticker_symbol)

    # Check if data is already in the cache
    if (
        cache_key in financial_cache
        and cache_key in cashflow_cache
        and cache_key in closing_prices_cache
    ):
        quarterly_financials = financial_cache[cache_key]
        quarterly_cashflow = cashflow_cache[cache_key]
        closing_prices = closing_prices_cache[cache_key]
        dividends = dividends_cache[cache_key]
    else:

        # Fetch quarterly financial data and transpose it
        quarterly_financials = ticker.quarterly_income_stmt.T
        quarterly_financials.index = pd.to_datetime(quarterly_financials.index).date
        quarterly_financials.sort_index(ascending=False, inplace=True)

        # Fetch quarterly cashflow data and transpose it
        quarterly_cashflow = ticker.quarterly_cashflow.T
        quarterly_cashflow.index = pd.to_datetime(quarterly_cashflow.index).date
        quarterly_cashflow.sort_index(ascending=False, inplace=True)

        # Ensure we have at least 4 quarters of data as of the target date
        quarterly_financials = quarterly_financials[
            quarterly_financials.index <= target_date
        ].head(8) 
        quarterly_cashflow = quarterly_cashflow[
            quarterly_cashflow.index <= target_date
        ].head(8)
        # Store in cache
        financial_cache[cache_key] = quarterly_financials
        cashflow_cache[cache_key] = quarterly_cashflow
        
        # Checks on financial data availalbiliy
        if len(quarterly_financials) < 4:  # no enough quarterly data
            return result

        # fetching price is costly so let's do only if needed.
        closing_prices, dividends = get_closing_price(ticker_symbol, target_date)
        quarterly_dates = quarterly_financials.index
        closing_prices = closing_prices.reindex(
            quarterly_dates, method="nearest", tolerance=pd.Timedelta("1D")
        ).ffill()
        closing_prices_cache[cache_key] = closing_prices
        dividends_cache[cache_key] = dividends

    # Calculate the average of the quarterly closing prices
    avg_quarterly_price = sum(closing_prices) / len(closing_prices)
    # Sum the EPS for the trailing twelve months
    trailing_eps = quarterly_financials["Diluted EPS"].sum()
    # Sum the net income for the trailing twelve months
    trailing_net_income = quarterly_financials["Net Income"].sum()
    # Comute the svg outstanding share for the trailing twelve months
    trailing_outstanding_shares = quarterly_financials[
        "Basic Average Shares"
    ].sum() / len(closing_prices)
    # Sum the dividends for the trailing twelve months trailing_dividends = dividends.sum()
    trailing_dividends = dividends.sum()

    # Compute Characterisitics indicators
    result[FIELD_EVAL_RATIO_PE] = (avg_quarterly_price / trailing_eps, TYPE_RATIO)
    result[FIELD_EVAL_RATIO_PCF] = (
        avg_quarterly_price
        / (
            quarterly_cashflow["Operating Cash Flow"]
            / quarterly_financials["Basic Average Shares"]
        ).sum(),
        TYPE_RATIO,
    )
    result[FIELD_EVAL_PER_SHARE_DILUTED_EPS] = (trailing_eps, TYPE_CASH)
    result[FIELD_EVAL_BASIC_AVG_SHARE] = (trailing_outstanding_shares, TYPE_AMOUNT)
    result[FIELD_EVAL_MARKET_CAP_TTM] = (
        (quarterly_financials["Basic Average Shares"] * closing_prices).sum() / 4,
        TYPE_CASH,
    )
    result[FIELD_EVAL_MARKET_NET_INCOME_TTM] = (trailing_net_income, TYPE_CASH)
    result[FIELD_EVAL_DIVIDEND_YIELD_TTM] = (
        trailing_dividends / avg_quarterly_price * 100,
        TYPE_RATIO,
    )

    # Add all other indicators from financials and cashflow as TTM sums
    # AND calculate YoY Growth (Current vs 4th Oldest in the set)
    for df in [quarterly_financials, quarterly_cashflow]:
        for col in df.columns:
            # Construct the key name for TTM
            col_clean = col.replace(' ', '_').replace('/', '_').replace('-', '_')
            key_name_ttm = f"{col_clean}_TTM"
            
            # TTM Calculation
            val_ttm = df[col].sum()
            result[key_name_ttm] = (val_ttm, TYPE_CASH)
            
            # YoY Calculation
            # Formula: (Latest - Oldest) / Oldest
            # df is sorted descending by date, so iloc[0] is latest, iloc[-1] is oldest
            if len(df[col]) >= 4: # Should be true given head(4) and checks
                latest = df[col].iloc[0]
                oldest = df[col].iloc[-1]
                
                key_name_yoy = f"{col_clean}_YOY"
                if oldest != 0 and pd.notna(oldest) and pd.notna(latest):
                    val_yoy = (latest - oldest) / abs(oldest) * 100
                    result[key_name_yoy] = (val_yoy, TYPE_RATIO)
                else:
                     result[key_name_yoy] = (np.nan, TYPE_RATIO)

    # Fetch and process Estimates
    # Attempt to fetch estimates from yf.Ticker attributes
    try:
        # Earnings Estimates
        if hasattr(ticker, "earnings_estimate") and ticker.earnings_estimate is not None:
             ee = ticker.earnings_estimate
             # Structure assumption: Index contains "0Q", "+1Q", "0Y", "+1Y" or similar
             # Columns usually include "Avg"
             # Let's handle common yfinance structure or fallback
             # If index is implied: 0Q, 1Q, 0Y, 1Y
             # Or if index is the periods.
             # User specified "earning_estimates" (mock name). 
             # I will try to map loosely if possible or check specific keys
             if "0q" in ee.index and "avg" in ee.columns: result[FIELD_EVAL_EST_EARNINGS_0Q] = (ee.loc["0q", "avg"], TYPE_CASH)
             if "+1q" in ee.index and "avg" in ee.columns: result[FIELD_EVAL_EST_EARNINGS_1Q] = (ee.loc["+1q", "avg"], TYPE_CASH)
             if "0y" in ee.index and "avg" in ee.columns: result[FIELD_EVAL_EST_EARNINGS_0Y] = (ee.loc["0y", "avg"], TYPE_CASH)
             if "+1y" in ee.index and "avg" in ee.columns: result[FIELD_EVAL_EST_EARNINGS_1Y] = (ee.loc["+1y", "avg"], TYPE_CASH)

        # Revenue Estimates
        if hasattr(ticker, "revenue_estimate") and ticker.revenue_estimate is not None:
             re = ticker.revenue_estimate
             if "0q" in re.index and "avg" in re.columns: result[FIELD_EVAL_EST_REVENUE_0Q] = (re.loc["0q", "avg"], TYPE_CASH)
             if "+1q" in re.index and "avg" in re.columns: result[FIELD_EVAL_EST_REVENUE_1Q] = (re.loc["+1q", "avg"], TYPE_CASH)
             if "0y" in re.index and "avg" in re.columns: result[FIELD_EVAL_EST_REVENUE_0Y] = (re.loc["0y", "avg"], TYPE_CASH)
             if "+1y" in re.index and "avg" in re.columns: result[FIELD_EVAL_EST_REVENUE_1Y] = (re.loc["+1y", "avg"], TYPE_CASH)

        # Growth Estimates
        if hasattr(ticker, "growth_estimates") and ticker.growth_estimates is not None:
             ge = ticker.growth_estimates
             # Structure assumption: Index is periods
             # Or Columns. Usually one column with values
             # Typical indices: "0Q", "+1Q", "0Y", "+1Y", "+5Y"
             # Column might be ticker symbol or "Stock"
             col_name = ge.columns[0] if not ge.empty else None
             if col_name:
                 if "0q" in ge.index: result[FIELD_EVAL_EST_GROWTH_0Q] = (ge.loc["0q", col_name], TYPE_RATIO)
                 if "+1q" in ge.index: result[FIELD_EVAL_EST_GROWTH_1Q] = (ge.loc["+1q", col_name], TYPE_RATIO)
                 if "0y" in ge.index: result[FIELD_EVAL_EST_GROWTH_0Y] = (ge.loc["0y", col_name], TYPE_RATIO)
                 if "+1y" in ge.index: result[FIELD_EVAL_EST_GROWTH_1Y] = (ge.loc["+1y", col_name], TYPE_RATIO)
                 if "LTG" in ge.index: result[FIELD_EVAL_EST_GROWTH_5Y] = (ge.loc["LTG", col_name], TYPE_RATIO)

    except Exception as e:
        # print(f"Error fetching estimates for {ticker_symbol}: {e}")
        pass


    return result


def compute_characteristics(hlds: pd.DataFrame, char_fields: list) -> pd.DataFrame:

    char_fields = [cf for cf in char_fields if cf in CHARACTERISTICS_FIELDS]
    if len(char_fields) == 0:
        print("No valid characteristic field provided.")
        return hlds
    if hlds.empty:
        return hlds

    # Initialize the new columns with NaN
    for indicator in char_fields:
        hlds[indicator] = pd.NA

    # Iterate over each row and update the hlds DataFrame
    for idx, row in hlds.iterrows():
        try:
            indicators = get_char_indicator(row["ticker"], row["date"], char_fields)
            for indicator, value in indicators.items():
                if (
                    indicator in char_fields
                ):  # TODO: this condition should be moved inside 'get_char_indicator' and used to do less calcs.
                    hlds.at[idx, indicator] = value[0]
        except Exception as e:
            print(
                f"Characteristics: error processing row {idx} with ticker {row['ticker']}: {e}"
            )
            continue

    return hlds
