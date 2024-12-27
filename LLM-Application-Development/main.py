import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())  # read local .env file

client = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    openai_api_key=os.environ['OPENAI_API_KEY'],
    temperature=0.0,
)

template_string = """What is {a} + {b}?"""
a = 198
b = 2.75

prompt_template = ChatPromptTemplate.from_template(template_string, input_types={"a": int, "b": float})
print(prompt_template)

prompt_messages = prompt_template.format_messages(a=a, b=b)

print(prompt_messages)

# Call the LLM to translate to the style of the customer message
llm_response = client.invoke(prompt_messages)
print("LLM Response -", llm_response.content)


