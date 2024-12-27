import os
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
GPT_MODEL = "gpt-3.5-turbo-0125"
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],  # This is the default and can be omitted
)


def get_completion(prompt, model=GPT_MODEL):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


print(get_completion("What is 2.3 + 1.75?"))
