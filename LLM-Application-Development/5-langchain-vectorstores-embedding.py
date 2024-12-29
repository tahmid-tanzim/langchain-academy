import warnings
import numpy as np

from uuid import uuid4
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

warnings.filterwarnings('ignore')
_ = load_dotenv(find_dotenv())  # read local .env file

# 1. PDFs Loader
# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("data/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("data/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("data/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("data/MachineLearning-Lecture03.pdf")
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

# View Loaded PDFs
for page in docs:
    print(f"page_content ({len(page.page_content)}) -", page.page_content[0:20])
    print("metadata -", page.metadata, end="\n\n")

# 2. Split PDFs text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", "(?<=\\. )", " ", ""],
    length_function=len
)

split_text = text_splitter.split_documents(docs)
print("split_text count -", len(split_text))

# 3. Embeddings
embedding = OpenAIEmbeddings()
sports1 = "cricket"
sports2 = "football"
sports3 = "rugby"
embedding1 = embedding.embed_query(sports1)
embedding2 = embedding.embed_query(sports2)
embedding3 = embedding.embed_query(sports3)
print(f"1. Embedding ({len(embedding1)}) -", embedding1)
print(f"2. Embedding ({len(embedding2)}) -", embedding2)
print(f"3. Embedding ({len(embedding3)}) -", embedding3)

print(f"Embedding Similarity ({sports1} vs {sports2}) -", np.dot(embedding1, embedding2))
print(f"Embedding Similarity ({sports1} vs {sports3}) -", np.dot(embedding1, embedding3))
print(f"Embedding Similarity ({sports2} vs {sports3}) -", np.dot(embedding2, embedding3))

# 4. Vector stores
vectordb = Chroma(
    collection_name="machine_learning_lecture",
    embedding_function=embedding,
    persist_directory='./vectordb',
)

uuids = [str(uuid4()) for _ in range(len(split_text))]
vectordb.add_documents(documents=split_text, ids=uuids)
vectordb.persist()

print("DB Count -", vectordb._collection.count())

# 5. Similarity Search
question = "what did they say about matlab?"
search_result = vectordb.similarity_search_with_score(question, k=5)

print(f"Similarity Search Score ({search_result[0][1]}) -", search_result[0][0].page_content)
print(f"\n\nSimilarity Search Score ({search_result[1][1]}) -", search_result[1][0].page_content)
