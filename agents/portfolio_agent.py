import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tools.portfolio_tools import analyze_portfolio

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
    timeout=30,
    max_retries=0
)

SYSTEM_PROMPT = """
You are an Indian equity portfolio analysis agent.

Your responsibilities:
- Extract stock names and quantities from user input.
- Convert unambiguous stock names into NSE ticker symbols.
- Ask for clarification if symbols or quantities are unclear.
- Use the analyze_portfolio tool for all calculations.
- Never invent stock prices or calculate metrics independently.
- Explain portfolio value, allocation and diversification.
- Explain concentration risk and effective holdings.
- Report the source and date of market prices.
- Clearly state that concentration risk is not overall risk.
- Support NSE equities and INR only.
- Do not recommend buying or selling specific securities.
- If the tool fails, explain the limitation.
"""

portfolio_agent = create_agent(
    model=model,
    tools=[analyze_portfolio],
    system_prompt=SYSTEM_PROMPT
)


def run_portfolio_agent(question: str) -> str:

    result = portfolio_agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    for message in result["messages"]:
        message.pretty_print()

    return result["messages"][-1].content


if __name__ == "__main__":
    question = input("Enter your portfolio details: ")
    print(run_portfolio_agent(question))