import os
import psycopg

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import (MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter) 

embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small"
)

POSTGRES_URL = os.environ["DATABASE_URL"]
PGVECTOR_URL = os.environ["PGVECTOR_URL"]

collection_name = "docs"

vector_store = PGVector(
    embeddings = embeddings,
    collection_name = collection_name,
    connection = PGVECTOR_URL,
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

def index_section (section_id:str)-> None:
   section = load_section(section_id)

   chunks = create_chunks(section)

   for chunk in chunks:
        print("=" * 60)
        print(chunk.page_content)
        print(chunk.metadata)

   vector_store.add_documents(chunks)

   print(f"Index {len(chunks)} chunks"
         f"from {section['document_slug']}/ {section['slug']}")


def load_section(section_id: str) -> dict:
    query = """SELECT
                s.id AS section_id,
                s.slug AS section_slug,
                s.title AS section_title,
                s.markdown_content,

                d.id AS document_id,
                d.slug AS document_slug,
                d.title AS document_title
            
            FROM document_sections s
            JOIN documents d
                ON d.id = s.document_id
            WHERE s.id = %s
    """

    with psycopg.connect(POSTGRES_URL) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (section_id,))
            row = cursor.fetchone()

            if row is None:
                raise ValueError(
                        f"section {section_id} does not exist"
                    )

            return {
                "id": str(row[0]),
                "slug": row[1],
                "title": row[2],
                "markdown_content": row[3],

                "document_id": str(row[4]),
                "document_slug": row[5],
                "document_title": row[6],
                }

def create_chunks(section: dict) -> list[Document]:
    heading_chunks = markdown_splitter.split_text(
        section["markdown_content"]
    )

    chunks = recursive_splitter.split_documents(
            heading_chunks
    )

    for index, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                    "document_id" :section["document_id"],
                    "document_slug": section["document_slug"],
                    "document_title": section["document_title"],

                    "section_id": section ["id"],
                    "section_slug": section["slug"],
                    "section_title": section["title"],

                    "chunk_index": index,
            }
        )
    return chunks
            
def main() -> None:
    print("Hello from singed-pipeline!")

    section_id = input("Section UUID:")
    index_section(section_id)
