import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

_ = load_dotenv(find_dotenv())  # read local .env file

client = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    openai_api_key=os.environ['OPENAI_API_KEY'],
    temperature=0.0,
)

# 2. Chat API : LangChain
template_string_1 = """What is {a} + {b}?"""
a = 198
b = 2.75

prompt_template_1 = ChatPromptTemplate.from_template(template_string_1, input_types={"a": int, "b": float})
prompt_messages_1 = prompt_template_1.format_messages(a=a, b=b)

# llm_response = client.invoke(prompt_messages_1)
# print(f"The output of {a} + {b} =", llm_response.content)

# 3. Output Parsers
template_string_2 = """Translate the English message to Bengali & French. \
message: {english_message}
{output_format_instructions}"""

english_message = "How are you doing?"
word_count_schema = ResponseSchema(
    name="word_count",
    description="Number of words in english message",
    type="Number"
)
bengali_schema = ResponseSchema(
    name="bengali",
    description="English to Bengali translated message",
    type="String"
)
french_schema = ResponseSchema(
    name="french",
    description="English to French translated message",
    type="String"
)

output_parser = StructuredOutputParser.from_response_schemas([
    bengali_schema,
    french_schema,
    word_count_schema,
])

output_format_instructions = output_parser.get_format_instructions()
prompt_template_2 = ChatPromptTemplate.from_template(template=template_string_2)

prompt_messages_2 = prompt_template_2.format_messages(
    english_message=english_message,
    output_format_instructions=output_format_instructions
)

llm_response = client.invoke(prompt_messages_2)
print("LLM Response -", llm_response.content)
output_dict = output_parser.parse(llm_response.content)
print("output_dict -", output_dict, type(output_dict))
print("Bengali -", output_dict.get('bengali'), type(output_dict.get('bengali')))
print("French -", output_dict.get('french'), type(output_dict.get('french')))
print("Word Count -", output_dict.get('word_count'), type(output_dict.get('word_count')))

