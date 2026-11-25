import math
from langchain_core.tools import tool
from services.market_data import get_historical_prices


@tool
def analyze_stock(
    symbol: str,
    period: str = "6mo"
) -> dict:
    """Analyze historical stock or index performance.

    Supported periods: 1mo, 3mo, 6mo, 1y, 2y, 5y.
    Use Yahoo Finance symbols such as TCS.NS
    or ^NSEI for Nifty 50.
    """

    symbol = symbol.strip().upper()

    history = get_historical_prices(symbol, period)

    prices = history["Close"]
    daily_returns = prices.pct_change().dropna()

    start_price = float(prices.iloc[0])
    end_price = float(prices.iloc[-1])

    # Historical return
    total_return = (
        end_price / start_price - 1
    ) * 100

    # Annualized volatility
    volatility = (
        float(daily_returns.std()) * math.sqrt(252) * 100
        if len(daily_returns) >= 2 else None
    )

    # Maximum drawdown
    running_max = prices.cummax()
    drawdown = prices / running_max - 1
    max_drawdown = abs(float(drawdown.min())) * 100

    # Moving averages
    ma_20 = (
        float(prices.tail(20).mean())
        if len(prices) >= 20 else None
    )

    ma_50 = (
        float(prices.tail(50).mean())
        if len(prices) >= 50 else None
    )

    # Simple trend classification
    if ma_20 is None or ma_50 is None:
        trend = "Insufficient data"
    elif end_price > ma_20 > ma_50:
        trend = "Bullish"
    elif end_price < ma_20 < ma_50:
        trend = "Bearish"
    else:
        trend = "Mixed"

    return {
        "symbol": symbol,
        "period": period,
        "start_date": str(history.index[0].date()),
        "end_date": str(history.index[-1].date()),
        "start_adjusted_price": round(start_price, 2),
        "end_adjusted_price": round(end_price, 2),
        "total_return_pct": round(total_return, 2),
        "annualized_volatility_pct":
            round(volatility, 2)
            if volatility is not None else None,
        "max_drawdown_pct": round(max_drawdown, 2),
        "ma_20": round(ma_20, 2)
            if ma_20 is not None else None,
        "ma_50": round(ma_50, 2)
            if ma_50 is not None else None,
        "trend": trend,
        "trading_days": len(prices),
        "source": "Yahoo Finance"
    }