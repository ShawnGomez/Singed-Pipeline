import os

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import (MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter) 

embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-large"
)

connection = os.environ["DATABASE_URL"] 

collection_name = "docs"

vector_store = PGVector(
    embeddings = embeddings,
    collection_name = collection_name,
    connection = connection,
    use_jsonb = True,
)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 150,
)

markdown_splitter = MarkdownHeaderTextSplitter(
headers_to_split_on = [
    ("##", "subsection"),
    ("###", "subsubsection"),
],
    strip_headers = False,
)

def index_section (section:dict)-> None:
    markdown_chunks = markdown_splitter.split_text(
            section["markdown_content"],
            )        
    chunks = recursive_splitter.split_documents(
                markdown_chunks
    )
    
    for chunk in chunks:
        chunk.metadata.update(
                {
                    "document_id" :section["document_id"],
                    "section_id": section ["id"],
                    "document_slug": section["document_slug"],
                    "section_slug": section["slug"],
                    "title": section["title"],
                    }
            )
        vector_store.add_documents(chunks)
            
def main() -> None:
    print("Hello from singed-pipeline!")
