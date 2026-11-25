import yfinance as yf


def get_stock_price(symbol: str) -> dict:
    """Fetch the latest available daily price."""

    symbol = symbol.strip().upper()

    if not symbol:
        raise ValueError("Stock symbol is required.")

    ticker = yf.Ticker(symbol)

    history = ticker.history(
        period="5d",
        interval="1d",
        auto_adjust=False
    )

    history = history.dropna(subset=["Close"])

    if history.empty:
        raise ValueError(
            f"No market data available for {symbol}"
        )

    latest = history.iloc[-1]

    return {
        "symbol": symbol,
        "price": round(float(latest["Close"]), 2),
        "date": str(history.index[-1].date()),
        "price_type": "latest_available_daily_price",
        "source": "Yahoo Finance"
    }

def get_historical_prices(
    symbol: str,
    period: str = "6mo"
):
    """Retrieve adjusted historical daily prices."""

    symbol = symbol.strip().upper()

    if not symbol:
        raise ValueError("Stock symbol is required.")

    allowed_periods = {
        "1mo", "3mo", "6mo",
        "1y", "2y", "5y"
    }

    if period not in allowed_periods:
        raise ValueError("Unsupported time period.")

    ticker = yf.Ticker(symbol)

    history = ticker.history(
        period=period,
        interval="1d",
        auto_adjust=True
    )

    history = history.dropna(subset=["Close"])

    if len(history) < 2:
        raise ValueError(
            f"Insufficient historical data for {symbol}"
        )

    return history