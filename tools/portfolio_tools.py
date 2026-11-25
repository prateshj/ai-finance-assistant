import math
from langchain_core.tools import tool
from services.market_data import get_stock_price


@tool
def analyze_portfolio(
    holdings: dict[str, float]
) -> dict:
    """Calculate stock portfolio value and allocation.

    Holdings must contain NSE stock symbols and
    their quantities.

    Example:
    {"RELIANCE.NS": 100, "TCS.NS": 50}

    Supports INR-denominated NSE equities only.
    """

    if not holdings:
        raise ValueError("Portfolio cannot be empty.")

    positions = []
    total_value = 0.0

    for symbol, quantity in holdings.items():

        symbol = symbol.strip().upper()

        if not symbol.endswith(".NS"):
            raise ValueError(
                f"Unsupported NSE symbol: {symbol}"
            )

        if not math.isfinite(quantity) or quantity <= 0:
            raise ValueError(
                f"Invalid quantity for {symbol}"
            )

        market_data = get_stock_price(symbol)
        price = market_data["price"]

        if not math.isfinite(price) or price <= 0:
            raise ValueError(
                f"Invalid market price for {symbol}"
            )

        value = price * quantity
        total_value += value

        positions.append({
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "price_date": market_data["date"],
            "market_value": value
        })

    for position in positions:
        position["allocation_pct"] = round(
            position["market_value"] / total_value * 100,
            2
        )

        position["market_value"] = round(
            position["market_value"], 2
        )

    largest_position = max(
        positions,
        key=lambda x: x["allocation_pct"]
    )

    # Calculate portfolio concentration
    weights = [
        p["allocation_pct"] / 100
        for p in positions
    ]

    hhi = sum(w ** 2 for w in weights)

    diversification_score = 100 * (1 - hhi)

    effective_holdings = 1 / hhi

    # Basic concentration risk assessment
    if hhi >= 0.50:
        concentration_risk = "High"
    elif hhi >= 0.25:
        concentration_risk = "Moderate"
    else:
        concentration_risk = "Low"

    return {
    "currency": "INR",
    "total_value": round(total_value, 2),
    "number_of_holdings": len(positions),
    "holdings": positions,
    "largest_holding": largest_position["symbol"],
    "largest_allocation_pct":
        largest_position["allocation_pct"],
    "concentration_index": round(hhi, 4),
    "diversification_score":
        round(diversification_score, 2),
    "effective_holdings":
        round(effective_holdings, 2),
    "concentration_risk": concentration_risk,
    "source": "Yahoo Finance"
    }