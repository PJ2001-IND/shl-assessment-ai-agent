import asyncio
import os
from models import ChatRequest, Message
from engine import process_chat

# Set mock API key if not set
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "mock_key")

async def test_agent():
    # Test 1: Vague query
    print("--- Test 1: Vague Query ---")
    req = ChatRequest(messages=[Message(role="user", content="I need an assessment.")])
    try:
        res = await process_chat(req)
        print("Vague Reply:", res.reply)
        print("Recommendations:", res.recommendations)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(test_agent())
