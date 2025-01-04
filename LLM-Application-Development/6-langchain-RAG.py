import warnings

from dotenv import load_dotenv, find_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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

# 3. Retrieval: Testing Similarity Search
question = "What are major topics for this class?"
docs = vectorDB.max_marginal_relevance_search(question, k=2, fetch_k=3)
print("Question:", question)
print("Search Result:")
print(f"\n{'-' * 100}\n".join([f"Document {i + 1}:\n\n" + d1.page_content for i, d1 in enumerate(docs)]))

# 4. Load LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 5. Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
 just say that you don't know, don't try to make up an answer. Use three sentences maximum. \
 Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
<{context}>
Question: <{question}>
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 6. RetrievalQA chain
# Inject top 2 retrieval from vector DB
# Default Chain Type
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorDB.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
response = qa_chain({"query": question})
print("LLM Response -", response["result"])
print("Source Documents -", response["source_documents"])

# 7. Chain type - map_reduce
# qa_chain_mr = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectorDB.as_retriever(),
#     chain_type="map_reduce"
# )
#
# qa_chain_mr = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectorDB.as_retriever(),
#     chain_type="refine"
# )
