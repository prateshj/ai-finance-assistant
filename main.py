
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],timeout=30.0, max_retries=1
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def ask_finance_assistant(question: str) -> str:

    response = client.responses.create(
        model=MODEL,
        instructions=(
            "You are a financial education assistant "
            "specializing in the Indian financial market. "
            "Explain financial concepts in simple language. "
            "Do not provide personalized investment advice."
        ),
        input=question
    )

    return response.output_text


if __name__ == "__main__":

    question = input("Ask a financial question: ")

    answer = ask_finance_assistant(question)

    print("\nFinance Assistant:")
    print(answer)
