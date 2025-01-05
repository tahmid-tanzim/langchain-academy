import warnings

from dotenv import load_dotenv, find_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

warnings.filterwarnings('ignore')
_ = load_dotenv(find_dotenv())  # read local .env file

# 1. Load vector stores & Embedding model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", )
vectorDB = Chroma(
    collection_name="machine_learning_lecture",
    embedding_function=embeddings_model,
    persist_directory='./vectorDB',
)

# 2. Test vector store loaded successfully
print(vectorDB._collection.count())

# 3. Load LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 5. Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
 just say that you don't know, don't try to make up an answer. Use three sentences maximum. \
 Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
<{context}>
Question: <{question}>
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 6. Memory to remember previous chat history
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


# 7. ConversationalRetrievalChain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorDB.as_retriever(),
    memory=memory,
    # chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "Is probability a class topic?"
response = qa_chain({"question": question})
print("\nLLM Response 1 -", response["answer"], end="\n")

question = "What are those prerequisites needed?"
response = qa_chain({"question": question})
print("\nLLM Response 2 -", response["answer"], end="\n")
