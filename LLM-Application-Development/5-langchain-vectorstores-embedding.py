import warnings
import numpy as np

from uuid import uuid4
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import SVMRetriever
from langchain_community.retrievers import TFIDFRetriever

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
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", )


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def sport_embeddings():
    sports1 = "cricket"
    sports2 = "football"
    sports3 = "soccer"

    embedding1 = embeddings_model.embed_query(sports1)
    embedding2 = embeddings_model.embed_query(sports2)
    embedding3 = embeddings_model.embed_query(sports3)

    print(f"1. Embedding ({len(embedding1)}) -", embedding1)
    print(f"2. Embedding ({len(embedding2)}) -", embedding2)
    print(f"3. Embedding ({len(embedding3)}) -", embedding3)

    print(f"Embedding Similarity ({sports1} vs {sports2}) -", cosine_similarity(embedding1, embedding2))
    print(f"Embedding Similarity ({sports1} vs {sports3}) -", cosine_similarity(embedding1, embedding3))
    print(f"Embedding Similarity ({sports2} vs {sports3}) -", cosine_similarity(embedding2, embedding3))


sport_embeddings()


# 4. Vector stores
vectorDB = Chroma(
    collection_name="machine_learning_lecture",
    embedding_function=embeddings_model,
    persist_directory='./vectorDB',
)

# uuids = [str(uuid4()) for _ in range(len(split_text))]
# vectorDB.add_documents(documents=split_text, ids=uuids)
# vectorDB.persist()

print("DB Count -", vectorDB._collection.count())

# 5. Retrieval: Similarity Search with Score
question = "what did they say about matlab?"

search_result = vectorDB.similarity_search_with_score(question, k=5)
print(f"Similarity Search Score ({search_result[0][1]}) -", search_result[0][0].page_content)
print(f"\n\nSimilarity Search Score ({search_result[1][1]}) -", search_result[1][0].page_content, end="\n\n")

# 6. Retrieval: MMR Search
search_result_2 = vectorDB.max_marginal_relevance_search(question, k=2, fetch_k=3)
print("1. MMR Search Score -", search_result_2[0].page_content)
print("\n\n2. MMR Search Score -", search_result_2[1].page_content, end="\n\n")

# 7. Retrieval: Similarity Search by metadata
search_result_3 = vectorDB.similarity_search(
    question,
    k=3,
    filter={"source": "data/MachineLearning-Lecture02.pdf"}
)
for d in search_result_3:
    print("Filter from Lecture 2 -", d.metadata, f"\n{d.page_content[:25]}...\n\n")

# 8. Retrieval: self-query retriever
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)


def test_self_query_retriever():
    """
    SelfQueryRetriever uses an LLM to extract:
        1. The query string to use for vector search
        2. A metadata filter to pass in as well
    """
    document_content_description = "Lecture notes"
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The lecture the chunk is from, should be one of "
                        "`data/MachineLearning-Lecture01.pdf`, "
                        "`data/MachineLearning-Lecture02.pdf`, or "
                        "`data/MachineLearning-Lecture03.pdf`",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="The page from the lecture",
            type="integer",
        ),
    ]

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorDB,
        document_content_description,
        metadata_field_info,
        verbose=True
    )

    question_2 = "what did they say about regression in the third lecture?"
    search_result_4 = retriever.get_relevant_documents(question_2)
    """
    search_result_4:
    query='regression' 
    filter=Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='source', value='data/MachineLearning-Lecture03.pdf') 
    limit=None
    """
    for r4 in search_result_4:
        print("Filter from Lecture 3 -", r4.metadata, f"\n{r4.page_content[:100]}...\n\n")


test_self_query_retriever()

# 9. Retrieval: Contextual Compression


def pretty_print_docs(docs2):
    print(f"\n{'-' * 100}\n".join([f"Document {i + 1}:\n\n" + d2.page_content for i, d2 in enumerate(docs2)]))


def test_contextual_compression():
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorDB.as_retriever(search_type="mmr")
    )
    question2 = "what did they say about matlab?"
    compressed_docs = compression_retriever.get_relevant_documents(question2)
    print("Contextual Compression -\n")
    pretty_print_docs(compressed_docs)


test_contextual_compression()

# 10. Retrieval: Without Embeddings / VectorDB
# Load PDF
loader = PyPDFLoader("data/MachineLearning-Lecture01.pdf")
pages = loader.load()
all_page_text = [p.page_content for p in pages]
joined_page_text = " ".join(all_page_text)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_text(joined_page_text)

# Retrieve
svm_retriever = SVMRetriever.from_texts(splits, embeddings_model)
question = "What are major topics for this class?"
docs_svm = svm_retriever.get_relevant_documents(question)
print("SVM Retriever -", docs_svm[0])

tfidf_retriever = TFIDFRetriever.from_texts(splits)
question = "what did they say about matlab?"
docs_tfidf = tfidf_retriever.get_relevant_documents(question)
print("TFIDF Retriever -", docs_tfidf[0])
