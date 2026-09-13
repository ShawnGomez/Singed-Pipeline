import os
import psycopg
import hashlib

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import (MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter) 

load_dotenv()

embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small"
)

POSTGRES_URL = os.environ["DATABASE_URL"]
PGVECTOR_URL = os.environ["PGVECTOR_URL"]


vector_store = PGVector(
    embeddings = embeddings,
    collection_name = "docs",
    connection = PGVECTOR_URL,
    use_jsonb = True,
)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 150,
    separators = ["\n\n", "\n", ".", " ", ""],
)

markdown_splitter = MarkdownHeaderTextSplitter(
headers_to_split_on = [
    ("#", "heading"),
    ("##", "subsection"),
    ("###", "subsubsection"),
],
    strip_headers = False,
)

def make_chunk_id(section_id: str, chunk_index: int) -> str:
    return f"{section_id}:{chunk_index}"


def index_section (section_id:str)-> None:
   section = load_section(section_id)
   chunks = create_chunks(section)

   chunk_ids = [
        make_chunk_id(section["id"], index)
        for index in range (len(chunks))
    ]

   vector_store.add_documents(documents = chunks, ids = chunk_ids)

   print(f"Index {len(chunks)} chunks"
         f"from {section['document_slug']}/ {section['slug']}")


def load_section(section_id: str) -> dict:
    query = """
        SELECT
                s.id,
                s.slug,
                s.title,
                s.markdown_content,

                d.id,
                d.slug,
                d.title
            
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

    indexed_chunks: list[Document] = []

    for chunk_index, chunk in enumerate(chunks):

        headings = [
            chunk.metadata.get("heading"),
            chunk.metadata.get("subsection"),
            chunk.metadata.get("subsubsection"),
        ]

        heading_path = " > ".join(
            heading for heading in headings if heading
        )

        prefix_parts = [
            f"Document: {section['document_title']}",
            f"Section: {section['title'] or section['slug']}",
        ]

        if heading_path:
            prefix_parts.append(f"Heading: {heading_path}")

        searchable_content = (
            "\n".join(prefix_parts)
            + "\n\n"
            + chunk.page_content.strip()
        )

        content_hash = hashlib.sha256(
            searchable_content.encode("utf-8")
        ).hexdigest()

        indexed_chunks.append(
            Document(
                page_content=searchable_content,
                metadata={
                    **chunk.metadata,
                    "document_id": section["document_id"],
                    "document_slug": section["document_slug"],
                    "document_title": section["document_title"],
                    "section_id": section["id"],
                    "section_slug": section["slug"],
                    "section_title": section["title"],
                    "chunk_index": chunk_index,
                    "content_hash": content_hash,
                },
            )
        )

    return indexed_chunks
            