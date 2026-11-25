import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools.finance_tools import finance_qa

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
    timeout=30,
    max_retries=0
)

SYSTEM_PROMPT = """
You are a specialized Indian financial education agent.

Your responsibilities:
- Explain financial concepts in simple language.
- Use the finance_qa tool for financial knowledge questions.
- Preserve source citations returned by the tool.
- Do not invent financial facts.
- Acknowledge insufficient information.
- Do not provide personalized investment advice.
"""

finance_agent = create_agent(
    model=model,
    tools=[finance_qa],
    system_prompt=SYSTEM_PROMPT
)


# def run_finance_agent(question: str) -> str:

#     result = finance_agent.invoke({
#         "messages": [
#             {"role": "user", "content": question}
#         ]
#     })

#     return result["messages"][-1].content

def run_finance_agent(question: str) -> str:

    result = finance_agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    # Inspect the agent execution trace
    for message in result["messages"]:
        message.pretty_print()

    return result["messages"][-1].content


if __name__ == "__main__":

    question = input("Enter your financial question: ")

    print(run_finance_agent(question))