import warnings
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader

warnings.filterwarnings('ignore')
_ = load_dotenv(find_dotenv())  # read local .env file

# 1. PDFs
loader = PyPDFLoader("data/MachineLearning-Lecture01.pdf")
pages = loader.load()

print(len(pages))
first_page = pages[-1]
print("page_content -", first_page.page_content)
print("metadata -", first_page.metadata)

# 2. URLs
loader = WebBaseLoader("https://raw.githubusercontent.com/pytorch/tutorials/refs/heads/main/README.md")
docs = loader.load()
print("PyTorch Tutorials -", docs[0].page_content)
