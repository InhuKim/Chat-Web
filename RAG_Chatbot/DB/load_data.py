import os
import argparse

from tqdm import tqdm

import chromadb
from chromadb.utils import embedding_functions

from langchain.document_loaders.csv_loader import CSVLoader


def main(
    documents_directory: str = "documents",
    collection_name: str = "documents_collection",
    persist_directory: str = ".",
) -> None:
    # Read all files in the data directory
    documents = []
    metadatas = []


    data = CSVLoader(documents_directory,
                     source_column='text',
                     metadata_columns=['label', 'search_query'])
    data = data.load()

    for value in data:
        temp_df = {}
        documents.append(value.metadata['source'])
        temp_df['label'] = value.metadata['label']
        temp_df['query'] = value.metadata['search_query']
        metadatas.append(temp_df)

    # Instantiate a persistent chroma client in the persist_directory.
    # Learn more at docs.trychroma.com
    client = chromadb.PersistentClient(path=persist_directory)

    # If the collection already exists, we just return it. This allows us to add more
    # data to an existing collection.
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-mpnet-base-v2")
    collection = client.get_or_create_collection(name=collection_name, embedding_function=sentence_transformer_ef)

    # Create ids from the current count
    count = collection.count()
    print(f"Collection already contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]

    # Load the documents in batches of 100
    for i in tqdm(range(len(documents))):
        collection.add(
            ids=ids[i],
            documents=documents[i],
            metadatas=metadatas[i]
        )

    new_count = collection.count()
    print(f"Added {new_count - count} documents")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load documents from a directory into a Chroma collection"
    )

    # Add arguments
    parser.add_argument(
        "--data_directory",
        type=str,
        default="/home/RAG_Chatbot/Data/db_input.csv",
        help="The directory where your text files are stored",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="documents_collection",
        help="The name of the Chroma collection",
    )
    parser.add_argument(
        "--persist_directory",
        type=str,
        default="/home/RAG_Chatbot/DB/chroma_storage",
        help="The directory where you want to store the Chroma collection",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        documents_directory=args.data_directory,
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
    )