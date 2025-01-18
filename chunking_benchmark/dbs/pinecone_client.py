import os
from dotenv import load_dotenv
import time

from pinecone import Pinecone, ServerlessSpec


class PineconeClient:
    def __init__(self):
        load_dotenv(override=True)
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = None
        # print(self.pc.list_indexes())

    def create_index(
        self,
        index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    ):
        if not self.pc.has_index(index_name):
            print(f"Creating index '{index_name}'...")
            self.pc.create_index(index_name, dimension=dimension, metric=metric, spec=spec)
        else:
            print(f"Index '{index_name}' already exists.")

        while not self.pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        self.index = self.pc.Index(index_name)

    def del_index(self, index_name):
        if self.pc.has_index(index_name):
            self.pc.delete_index(index_name)
            print(f"Index '{index_name}' deleted.")
        else:
            print(f"Index '{index_name}' does not exist.")

    def create_namespace_in_index(self, index_name, namespace):
        self.index = self.pc.Index(index_name)
        if namespace in self.index.describe_index_stats()["namespaces"]:
            print(f"Namespace '{namespace}' already exists in index, overiding '{index_name}'.")
            self.del_namespace_in_index(index_name, namespace)

    def del_namespace_in_index(self, index_name, namespace):
        self.index = self.pc.Index(index_name)
        self.index.delete(delete_all=True, namespace=namespace)
        print(f"Namespace '{namespace}' deleted from index '{index_name}'.")

    def upsert_to(self, index_name, vectors, namespace=None):
        self.index = self.pc.Index(index_name)
        self.index.upsert(vectors=vectors, namespace=namespace)
        print(f"Vectors upserted to index '{index_name}' in namespace '{namespace}'.")

    def upsert_in_batches(self, index_name, vectors, namespace=None, batch_size=100):
        self.index = self.pc.Index(index_name)
        total_vectors = len(vectors)
        for i in range(0, total_vectors, batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            print(f"Batch {i // batch_size + 1} upserted, size: {len(batch)}")
        print(f"Total of {total_vectors} vectors upserted in batches of {batch_size}.")

    def query(self, index_name, vector, top_k, filter={}, namespace=None, include_metadata=True):
        self.index = self.pc.Index(index_name)
        result = self.index.query(vector=vector, filter=filter, top_k=top_k, namespace=namespace, include_metadata=include_metadata)
        return result
