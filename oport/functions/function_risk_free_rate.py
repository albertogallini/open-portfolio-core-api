# %%
import yfinance as yf


def get_risk_free_rate(ccy):
    """Gets the risk-free rate for a given currency.

    Args:
        ccy: The 3-letter currency code (e.g., 'USD', 'EUR', 'JPY').

    Returns:
        float: The risk-free rate.
        str: The ticker used to fetch the risk-free rate.
    """

    # Mapping of currencies to their respective ticker symbols for risk-free rates.
    ticker_mapping = {
        "USD": "^IRX",  # 3-Month US Treasury Bill
        "EUR": "EU01",  # 3-Month Euribor
        "JPY": "JP01",  # 3-Month TIBOR
        "GBP": "UK01",  # 3-Month LIBOR
        "AUD": "AU01",  # 3-Month BBSW
        "CAD": "CA01",  # 3-Month CDOR
        "CHF": "CH01",  # 3-Month SARON
        "CNY": "CN01",  # 3-Month SHIBOR
        "SEK": "SE01",  # 3-Month STIBOR
        "NOK": "NO01",  # 3-Month NIBOR
        "DKK": "DK01",  # 3-Month CIBOR
        "INR": "IN01",  # 3-Month MIBOR
        "BRL": "BR01",  # 3-Month CDI
        "MXN": "MX01",  # 3-Month TIIE
        "ZAR": "ZA01",  # 3-Month JIBAR
        "RUB": "RU01",  # 3-Month RUONIA
        "HKD": "HK01",  # 3-Month HIBOR
        "SGD": "SG01",  # 3-Month SIBOR
        "NZD": "NZ01",  # 3-Month NZBORA
    }

    if ccy not in ticker_mapping:
        return None, None

    ticker = ticker_mapping[ccy]
    tickerData = yf.Ticker(ticker)
    try:
        # Get the history of the ticker
        history = tickerData.history(
            period="5d", interval="1d", auto_adjust=True, back_adjust=True
        )

        # Check if the history is empty
        if history.empty:
            return None, ticker

        # Return the latest closing price as the risk-free rate
        risk_free_rate = history["Close"][-1]
        return risk_free_rate, ticker
    except Exception as e:
        print(f"Error getting risk-free rate for {ccy} ({ticker}): {e}")
        return None, ticker


# %%
currencies = [
    "USD",
    "EUR",
    "JPY",
    "GBP",
    "AUD",
    "CAD",
    "CHF",
    "CNY",
    "SEK",
    "NOK",
    "DKK",
    "INR",
    "BRL",
    "MXN",
    "ZAR",
    "RUB",
    "HKD",
    "SGD",
    "NZD",
]  # Add more currencies here

for ccy in currencies:
    rate, ticker = get_risk_free_rate(ccy)
    if rate is not None:
        print(f"Risk-free rate for {ccy} ({ticker}): {rate}")
    else:
        print(f"Could not retrieve risk-free rate for {ccy}")

# %%
