import uuid
import shutil
import os
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def create_vectorstore(chunks, embedding_function, vectorstore_path):
    # Create a list of unique ids for each document based on the content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]

    # Ensure that only unique docs with unique ids are kept
    unique_ids = set()
    unique_chunks = []

    for chunk, id in zip(chunks, ids):
        if id not in unique_ids:
            unique_ids.add(id)
            unique_chunks.append(chunk)

    # Create a new Chroma database from the documents
    vectorstore = Chroma.from_documents(documents=unique_chunks,
                                        ids=list(unique_ids),
                                        embedding=embedding_function,
                                        persist_directory=vectorstore_path)

    return vectorstore


# Create embeddings
def get_embedding_function(OPENAI_API_KEY):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY
    )
    return embeddings