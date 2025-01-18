import boto3
import json
from dotenv import load_dotenv

from .base_reranker import BaseReranker


class AWSReranker(BaseReranker):
    def __init__(
        self,
        region_name="us-east-1",
        endpoint_name="Endpoint-Cohere-Rerank-2-Model-English-1",
        **kwargs,
    ):
        super().__init__(name="aws-reranker")
        load_dotenv(override=True)
        self.client = boto3.client("sagemaker-runtime", region_name=region_name)
        self.endpoint_name = endpoint_name

    def _rerank(
        self,
        res,
        query,
        top_n=3,
        model="rerank-english-v3.0",
        max_chunks_per_doc=1,
        return_documents=True,
        **kwargs,
    ):
        input_data = {
            "query": query,
            "documents": res,
            "top_n": top_n,
            "return_documents": True,
            "max_chunks_per_doc": max_chunks_per_doc,
        }
        input_data = json.dumps(input_data, indent=4).encode("utf-8")
        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name, ContentType="application/json", Body=input_data
            )
            return json.loads(response["Body"].read().decode())
        except Exception as e:
            print(f"Error during ranking {self.name}: {e}")
            return {}


if __name__ == "__main__":
    reranker = AWSReranker()
    query = "What is the capital of the United States?"
    docs = [
        "Carson City is the capital city of the American state of Nevada. At the 2010 United States Census, Carson City had a population of 55,274.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
    ]
    reranked = reranker._rerank(res=docs, query=query)
    print(f"reranked: {reranked}")
