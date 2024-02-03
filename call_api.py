import sys
sys.path.append(".")
from models.api_base import APIBase
from logger import get_logger
logger = get_logger(__name__)

def call_api(prompt,temperature=0,engine="gpt-3.5-turbo-1106",max_tokens=1000):
    client=APIBase(
            engine=engine,
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0,
            best_of=1,
            api_key="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" ,
            organization_key="org-BRY8JpM4YwZisiyvCfwlNXcA"
        )
    result=client.get_multiple_sample(prompt)
    return result
