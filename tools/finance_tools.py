from langchain_core.tools import tool
from answer import generate_answer


@tool
def finance_qa(question: str) -> str:
    """Answer financial education questions using
    the indexed financial knowledge base.

    Use this tool for questions about financial
    concepts, mutual funds, ETFs, and other topics
    covered by the knowledge base.
    """
    return generate_answer(question)