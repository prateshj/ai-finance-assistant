from langchain_core.tools import tool
import math


@tool
def calculate_sip(
    monthly_investment: float,
    annual_return: float,
    years: int
) -> dict:
    """Calculate the projected future value of a monthly SIP.

    annual_return is a percentage, e.g. 10 for 10%.
    Assumes contributions at the end of each month.
    """

    if monthly_investment <= 0 or years <= 0:
        raise ValueError("Investment and years must be positive.")

    if not math.isfinite(annual_return) or annual_return < 0:
        raise ValueError("Return must be finite and non-negative.")

    r = annual_return / 1200
    n = years * 12

    invested = monthly_investment * n

    if r == 0:
        future_value = invested
    else:
        future_value = monthly_investment * (
            ((1 + r) ** n - 1) / r
        )

    return {
        "total_invested": round(invested, 2),
        "future_value": round(future_value, 2),
        "estimated_gain": round(future_value - invested, 2)
    }


@tool
def calculate_future_goal(
    current_cost: float,
    inflation_rate: float,
    years: int
) -> dict:
    """Calculate the future cost of a financial goal
    after accounting for annual inflation.

    inflation_rate is a percentage, e.g. 6 for 6%.
    """

    if current_cost <= 0 or years < 0:
        raise ValueError("Invalid cost or years.")

    if not math.isfinite(inflation_rate) or inflation_rate < 0:
        raise ValueError("Invalid inflation rate.")

    future_cost = current_cost * (
        1 + inflation_rate / 100
    ) ** years

    return {
        "current_cost": current_cost,
        "future_cost": round(future_cost, 2),
        "inflation_impact": round(
            future_cost - current_cost, 2
        )
    }

@tool
def calculate_step_up_sip(
    monthly_investment: float,
    annual_return: float,
    annual_step_up: float,
    years: int
) -> dict:
    """Calculate SIP maturity with an annual increase
    in monthly contributions.

    All rates are percentages, e.g. 12 for 12%.
    Contributions occur at the end of each month.
    The SIP amount increases after every 12 months.
    """

    values = [
        monthly_investment,
        annual_return,
        annual_step_up
    ]

    if not all(math.isfinite(v) for v in values):
        raise ValueError("Inputs must be finite.")

    if monthly_investment <= 0 or years <= 0:
        raise ValueError("Invalid investment or duration.")

    if annual_return < 0 or annual_step_up < 0:
        raise ValueError("Rates cannot be negative.")

    monthly_rate = annual_return / 1200

    corpus = 0.0
    total_invested = 0.0
    current_sip = monthly_investment

    for year in range(years):

        for month in range(12):
            corpus = corpus * (1 + monthly_rate)
            corpus += current_sip
            total_invested += current_sip

        current_sip *= (1 + annual_step_up / 100)

    return {
        "total_invested": round(total_invested, 2),
        "future_value": round(corpus, 2),
        "estimated_gain": round(
            corpus - total_invested, 2
        )
    }