import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools.goal_tools import (
    calculate_sip,
    calculate_future_goal,
    calculate_step_up_sip
)

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
    timeout=30,
    max_retries=0
)

SYSTEM_PROMPT = """
You are a financial goal planning agent.

- Use available tools for all calculations.
- Never perform calculations independently.
- Ask for missing required inputs.
- Explain results in simple language.
- Present monetary values in Indian rupees.
- Clearly state assumptions.
- Explain that projected returns are not guaranteed.
- Do not recommend specific investment products.
"""

goal_agent = create_agent(
    model=model,
    tools=[
        calculate_sip,
        calculate_future_goal,
        calculate_step_up_sip
    ],
    system_prompt=SYSTEM_PROMPT
)


def run_goal_agent(question: str) -> str:

    result = goal_agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    for message in result["messages"]:
        message.pretty_print()

    return result["messages"][-1].content


if __name__ == "__main__":
    question = input("Enter your financial goal: ")
    print(run_goal_agent(question))