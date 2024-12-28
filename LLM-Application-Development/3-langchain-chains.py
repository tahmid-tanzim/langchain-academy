import os
import warnings
import pandas as pd
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings('ignore')
_ = load_dotenv(find_dotenv())  # read local .env file

client = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    api_key=os.environ['OPENAI_API_KEY'],
    temperature=0.9,
    max_retries=2,
)

# 1. LLMChain
prompt_template = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?",
    input_types={"product": str}
)

chain = LLMChain(llm=client, prompt=prompt_template, output_parser=StrOutputParser())

product = "Waterproof Phone Pouch"
# response = chain.run(product)
# print("LLM Chain Output -", response)

# 2. SequentialChain
# prompt template 1: translate to english
first_prompt = ChatPromptTemplate.from_template("Translate the text to english: <{review}>")
chain_one = LLMChain(
    llm=client,
    prompt=first_prompt,
    output_key="english_review",
    output_parser=StrOutputParser()
)

# prompt template 2: summarize the english review
second_prompt = ChatPromptTemplate.from_template("Summarize the following text in 1 sentence: <{english_review}>")
chain_two = LLMChain(
    llm=client,
    prompt=second_prompt,
    output_key="english_summary",
    output_parser=StrOutputParser()
)

# prompt template 3: Find the default review language
third_prompt = ChatPromptTemplate.from_template("What language is the following text: <{review}>")
chain_three = LLMChain(
    llm=client,
    prompt=third_prompt,
    output_key="default_language",
    output_parser=StrOutputParser()
)

# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following summary in the specified languages:"
    "\nSummary: <{english_summary}>"
    "\nLanguages: ['{default_language}', 'Bengali']"
)

chain_four = LLMChain(
    llm=client,
    prompt=fourth_prompt,
    output_key="followup_messages"
)

combined_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["review"],
    output_variables=["english_review", "english_summary", "default_language", "followup_messages"],
    verbose=True
)

df = pd.read_csv('data.csv')

review = df.Review[5]
response = combined_chain(review)
print("LLM SequentialChain Output -", response)
