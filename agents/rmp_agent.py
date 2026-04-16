from openai import OpenAI
from data_models import RMPData
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a quantitative data extractor. Analyze the following student reviews and extract the core metrics. Rate your confidence lower if there are very few reviews or conflicting opinions.
"""

def run_rmp_agent(raw_scraped_reviews: str) -> RMPData:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract metrics from these reviews: {raw_scraped_reviews}"},
        ],
        response_format=RMPData
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Failed to parse RMPData from model response")

    return parsed   
