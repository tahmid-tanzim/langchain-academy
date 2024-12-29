import warnings

from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter

warnings.filterwarnings('ignore')
_ = load_dotenv(find_dotenv())  # read local .env file

# 1. PDFs Loader
loader = PyPDFLoader("data/MachineLearning-Lecture01.pdf")
pages = loader.load()

print(len(pages))
last_page = pages[-1]
print("page_content -", last_page.page_content)
print("metadata -", last_page.metadata)


# 2. URLs Loader
loader = WebBaseLoader(web_paths=["https://raw.githubusercontent.com/pytorch/tutorials/refs/heads/main/README.md"])
docs = loader.load()
print("PyTorch Tutorials -", docs[0].page_content)

# 3. Recursive splitting
some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentences. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator=" "
)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=5,
    separators=["\n\n", "\n", "(?<=\\. )", " ", ""]
)

c_text = c_splitter.split_text(some_text)
print("CharacterTextSplitter -")
for c in c_text:
    print('\t', len(c), c)
r_text = r_splitter.split_text(some_text)
print("RecursiveCharacterTextSplitter -")
for r in r_text:
    print('\t', len(r), r)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n"],
    length_function=len
)
split_texts = text_splitter.split_documents(pages)
print("Split Text in PDF -")
for t in split_texts[:10]:
    print(
        f'Page Content: ({len(t.page_content)})\n',
        t.page_content,
        "\nMetadata:",
        t.metadata,
        end=f"\n---------------------------------\n"
    )

# 4. Token splitting
# Tokens are often ~4 characters.
token_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
print(token_splitter.split_text("foo bar bazzyfoo"))

token_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=5)
split_texts_2 = token_splitter.split_documents(pages)
print("Split Text in PDF with TokenSplitter -")
for t in split_texts_2[:10]:
    print(
        f'Page Content: ({len(t.page_content)})\n',
        t.page_content,
        "\nMetadata:",
        t.metadata,
        end=f"\n---------------------------------\n"
    )
