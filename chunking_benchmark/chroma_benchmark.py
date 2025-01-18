import argparse
import joblib
import json
import logging
import os
import pandas as pd
import ast
from tqdm import tqdm
from typing import List, Dict, Any, Literal, get_args
from dotenv import load_dotenv
from datasets import Dataset
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import asyncio
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample

from schema import ChunkingMethod, VectorDB, EmbeddingModel
from retrievers.embedder import OpenAIEmbedder
from dbs.pinecone_client import PineconeClient
from retrievers.reranker import CohereReranker
from retrievers.aws_reranker import AWSReranker
from utils.retrievers import format_vectors, read_chunk_from_json
from utils.text import clean_text
from llms.llm import LLM
from utils.tokens import get_num_tokens, get_tokens_from_text, get_text_from_tokens
from utils.cache_decorator import cache_to_file


logging.basicConfig(level=getattr(logging, os.getenv("LOGLEVEL", "INFO").upper()))


def chunk_text(text: str, chunking_method: ChunkingMethod) -> List[Dict[str, str]]:
    raise NotImplementedError


def indexing(
    chunks: List[Dict[str, str]],
    embedding_model: EmbeddingModel = OpenAIEmbedder,
    db: VectorDB = VectorDB.PINECONE,
    index_name: str = "test-index-chunking-benchmark-chroma",
    namespace: str = "test-namespace-chunking-benchmark",
    upsert_batch_size: int = 100,
    **kwargs,
):
    logging.info(f"Indexing {len(chunks)} chunks using {embedding_model}, save to {db.value}")

    # Generate embedding vectors
    embedder = embedding_model(show_progress=True)
    embeddings = embedder.embed_chunk_s([chunk["text"] for chunk in chunks])
    vectors = format_vectors(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        metadatas=chunks,
    )

    # Upsert embeddings to db
    if not len(embeddings):
        logging.info("No embeddings generated.")
        return
    db_client = PineconeClient()
    dimension = len(embeddings[0])
    db_client.create_index(index_name=index_name, dimension=dimension)
    db_client.create_namespace_in_index(index_name, namespace)
    db_client.upsert_in_batches(index_name, vectors, namespace=namespace, batch_size=upsert_batch_size)
    # db_client.del_index(index_name)  # skip index deletion during testing if wanted

RerankMethod = Literal["aws", "cohere"]

@cache_to_file(location="./cache_dir", verbose=False)
def retrieve_relevant_chunks(
    question: str,
    db: VectorDB,
    embedder,
    db_client,
    index_name: str,
    namespace: str,
    retriever_topk=1000,
    rerank_topk=3,
    filter={},
    to_rerank: bool = True,
    rerank_method: RerankMethod = "aws",
    **kwargs,
) -> List[str]:
    logging.info(f"retriever_topk {retriever_topk}, rerank_topk {rerank_topk}")

    # retrieve topk chunks
    question_embedding = embedder.embed_chunk(question)
    vectors = db_client.query(
        index_name, vector=question_embedding, top_k=retriever_topk, filter=filter, namespace=namespace
    )

    if to_rerank:
        reranker = AWSReranker() if rerank_method == "aws" else CohereReranker()
        logging.info(f"Using reranker {rerank_method}")
        reranked_docs = reranker.rerank(
            query=question,
            res=vectors,
            top_n=rerank_topk,
        )
        docs = [clean_text(doc) for doc in reranked_docs]
        logging.info(f"Using reranker, retrieved {len(docs)} docs")
    else:
        matches = vectors.get("matches", [])
        docs = [match["metadata"]["text"] for match in matches]
        docs = [clean_text(doc) for doc in docs]
        logging.info(f"Retrieved {len(docs)} docs without reranker")
    return docs


def get_dataset(path_questions: str = "assets/chroma/questions.csv", num_questions: int = 2) -> pd.DataFrame:
    # chroma dataset: question,references,corpus_id
    df = pd.read_csv(path_questions, nrows=num_questions)
    df = df.rename(columns={"question": "Question", "references": "References"})
    df["reference_contexts"] = df.References.apply(lambda x: [_.get("content") for _ in ast.literal_eval(x)])
    df.References = df.References.apply(lambda x: " ".join([_.get("content") for _ in ast.literal_eval(x)]))
    return df


def eval_ragas_020(
    data: Dict[str, str],
    metric: List[Any] = [context_precision, context_recall],
    run_config=RunConfig(max_workers=4),
    **kwargs,
):
    """Evaluate the dataset using the new format in RAGAS 0.2.0."""
    # format dataset
    samples = []
    for question, ref, retrieved_context, reference_context in zip(
        data["questions"], data["references"], data["retrieved_contexts"], data["reference_contexts"]
    ):
        sample = SingleTurnSample(
            user_input=question,
            reference=ref,
            retrieved_contexts=[retrieved_context],
            reference_contexts=reference_context,
        )
        samples.append(sample)
    dataset = EvaluationDataset(samples=samples)

    # eval
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=42)
    # llm = ChatOpenAI(model="gpt-4o", temperature=0, seed=42)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    res = evaluate(dataset, metrics=metric, llm=llm, embeddings=embeddings, run_config=run_config)
    return res


async def run_single_benchmark_async(**kwargs):
    """Support func to run run_single_benchmark() in async mode"""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, lambda: run_single_benchmark(**kwargs))
    except Exception as e:
        logging.error(f"Error: {e}")
        result = {}
    return result


def get_context(chunks: List[str], token_limit: int = 1000):
    """Agreegate context from chunks upto token_limit."""
    context = ""
    counter, i = 0, 0
    for i, chunk in enumerate(chunks):
        num_token = get_num_tokens(chunk)
        if counter + num_token > token_limit:
            context += " \n\n " + get_text_from_tokens(get_tokens_from_text(chunk)[: (token_limit - counter)])
            counter = token_limit
            logging.debug(f"Cut off at {token_limit}")
            break
        counter += num_token
        context += " \n\n " + chunk
    logging.info(
        f"context {get_num_tokens(context)} tokens, counter: {counter}/{token_limit} tokens {i}/{len(chunks)} chunks"
    )
    return context


def run_single_benchmark(
    text: str,
    chunking_method: ChunkingMethod,
    vector_db: VectorDB,
    metrics: List[str],
    embedding_model: EmbeddingModel = OpenAIEmbedder,
    load_chunk_from_file: str = None,
    retriever_topk: int = 400,
    rerank_topk: int = 100,
    token_limit: int = 1200,
    num_questions: int = None,
    num_chunks: int = None,
    to_rerank: bool = True,
    path_questions: str = "assets/chroma/questions.csv",
    index_name: str = "test-index-chunking-benchmark-chroma",
    namespace: str = "test-namespace-chunking-benchmark",
    res_dir: str = "assets/the-fourth-wing/results",
    run_indexing: bool = True,
    run_reranking: bool = True,
    run_eval: bool = True,
    rerank_method: RerankMethod = "aws",
    **kwargs,
) -> Dict[str, float]:
    """Benchmark chunking strategy on a given eval dataset.

    Args:
        text (str): Text be be chunked
        chunking_method (ChunkingMethod): chunking strategy
        vector_db (VectorDb): The vector db to store the chunks
        metrics (List[str]): List of RAG metrics, RAGAS ones
        embedding_model (EmbeddingModel, optional): Embedding model. Defaults to OpenAIEmbedder.
        load_chunk_from_file (str, optional): path to the chunk file. Defaults to None.
        retriever_topk (int, optional): topk retrieval from the vector db. Defaults to 10.
        rerank_topk (int, optional): topk rerank. Defaults to 5.
        token_limit (int, optional): max number of token for the retrieved context. Defaults to 1200.
        num_questions (int, optional): number of question to eval. Defaults to None.
        num_chunks (int, optional): number of chunks to be indexed. Defaults to None.
        to_rerank (bool, optional): whether to use reranking during retrieval. Defaults to True.
        index_name (str, optional): vector db index name. Defaults to "test-index-chunking-benchmark-chroma".
        namespace (str, optional): vector db namespace. Defaults to "test-namespace-chunking-benchmark".
        res_dir (str, optional): dir to store result locally. Defaults to "assets/the-fourth-wing/results".
        run_indexing (bool, optional): trigger running indexing step. Defaults to True.
        run_reranking (bool, optional): trigger running rereanking step. Defaults to True.
        run_eval (bool, optional): trigger running eval step. Defaults to True.
        rerank_method (RerankMethod, optional): rerank method. Defaults to "aws".

    Returns:
        Dict[str, float]: the evaluation result.
    """

    # Chunk document
    if load_chunk_from_file:
        suffix = os.path.basename(load_chunk_from_file)
        chunks = read_chunk_from_json(load_chunk_from_file, num_chunks)
    else:
        suffix = chunking_method.value
        chunks = chunk_text(text, chunking_method)[:num_chunks]

    # Index chunks into vector embedder: database
    if run_indexing:
        indexing(
            chunks=chunks,
            db=vector_db,
            index_name=index_name,
            namespace=namespace,
            embedding_model=embedding_model,
        )

    # init params dataset, clients, embedder
    df = get_dataset(path_questions=path_questions, num_questions=num_questions)
    embedder = embedding_model()
    db_client = PineconeClient()
    result_group = {}

    # process by `corpus_id`, i.e., document-level
    if run_reranking:
        for corpus_id, df in df.groupby("corpus_id"):
            logging.info(f"suffix: {suffix}, corpus_id: {corpus_id}, num_questions: {len(df)}")

            # retrieve and cache
            contexts, references = [], []
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{suffix}: corpus_id {corpus_id}: len {len(df)}"):
                question = row["Question"]
                relevant_chunks = retrieve_relevant_chunks(
                    question,
                    vector_db,
                    index_name=index_name,
                    namespace=namespace,
                    embedder=embedder,
                    db_client=db_client,
                    retriever_topk=retriever_topk,
                    rerank_topk=rerank_topk,
                    filter={"chapter_name": row["corpus_id"]},
                    to_rerank=to_rerank,
                    rerank_method=rerank_method,
                )
                context = get_context(relevant_chunks, token_limit=token_limit)
                contexts.append(context)
                references.append(row["References"])

            if not run_eval:
                continue

            # eval
            data_dict = {
                "questions": df["Question"].tolist(),
                "references": df["References"].tolist(),
                "retrieved_contexts": contexts,
                "reference_contexts": df["reference_contexts"].tolist(),
            }
            with open(f"{res_dir}/{suffix}_data_dict-tokenlimit-{token_limit}-corpusid-{corpus_id}.pkl", "wb") as f:
                pickle.dump(data_dict, f)
            result = eval_ragas_020(data_dict, metric=metrics, run_config=RunConfig(max_workers=4))
            result_group[corpus_id] = {
                "scores": result._repr_dict,
                "scrores_raw": result._scores_dict,
                "len": len(df),
                "token_limit": token_limit,
                "context_len_char": [len(_) for _ in contexts],
            }
            logging.info(result_group[corpus_id])
            with open(f"{res_dir}/{suffix}_result_group-tokenlimit-{token_limit}-corpusid-{corpus_id}.json", "w") as f:
                json.dump(result_group[corpus_id], f, indent=4)

    print(f"FLAG: return {agreegate_doc_level_metric(result_group)}")
    return agreegate_doc_level_metric(result_group)


def agreegate_doc_level_metric(result_group: dict):
    if not len(result_group) or not sum([len(v) for v in result_group.values()]):
        return result_group
    res = {
        "scores": {},
        "scrores_raw": {"context_precision": [], "context_recall": []},
        "len": 0,
        "token_limit": [],
        "context_len_char": [],
        "chapter_name": [],
    }
    for k, v in result_group.items():
        if sum(v["context_len_char"]) == 0:
            # skip empty corpus/document
            logging.error(f"Agreegate: skip empty corpus {k}")
            continue
        res["scrores_raw"]["context_precision"].extend(v["scrores_raw"]["context_precision"])
        res["scrores_raw"]["context_recall"].extend(v["scrores_raw"]["context_recall"])
        res["len"] += v["len"]
        res["token_limit"].append(v["token_limit"])
        res["context_len_char"].extend(v["context_len_char"])
        res["chapter_name"].append(k)
    res["scores"]["context_precision"] = sum(res["scrores_raw"]["context_precision"]) / len(
        res["scrores_raw"]["context_precision"]
    )
    res["scores"]["context_recall"] = sum(res["scrores_raw"]["context_recall"]) / len(
        res["scrores_raw"]["context_recall"]
    )
    return {"all": res, **result_group}


def list_chunk_methods(chunk_dir):
    chunk_files = os.listdir(chunk_dir)
    chunk_files = [f for f in chunk_files if f.endswith(".json") and not f.endswith("_metadata.json")]
    return chunk_files


def analyse_results(res, res_dir: str = "assets/the-fourth-wing/results"):
    if not len(res) or not sum([len(v) for v in res.values()]):
        return None
    for k, v in res.items():
        print(f"{k}: {v}")
        res[k]["chunk_avg_size"] = int(k.split("-")[1])
        res[k]["method"] = k.split("-")[0]
    df = pd.DataFrame.from_dict(res, orient="index").reset_index(drop=True).sort_values(by="chunk_avg_size")

    plt.figure(figsize=(12, 12))
    metrics = ["context_precision", "context_recall", "accuracy"]
    titles = [
        "Context Precision vs Chunk Average Size",
        "Context Recall vs Chunk Average Size",
        "Accuracy vs Chunk Average Size",
    ]

    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i + 1)
        if metric not in df.columns:
            continue
        for method in df["method"].unique():
            subset = df[df["method"] == method]
            plt.plot(subset["chunk_avg_size"], subset[metric], marker="o", label=method)
        plt.title(titles[i])
        plt.xlabel("Chunk Average Size")
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend(loc="upper right")
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{res_dir}/chunk_avg_size_vs_metrics.png")


async def main(args):
    res = {}
    index_name = args.index_name
    chunk_dir = args.chunk_dir
    assert args.rerank_method in get_args(RerankMethod), f"Rerank method must be one of {get_args(RerankMethod)}"
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"{args.out_dir}/{date_str}"
    os.makedirs(out_dir, exist_ok=True)
    chunk_methods = list_chunk_methods(chunk_dir)
    print(chunk_methods)
    input("Press any key to continue...")

    tasks, suffixes = [], []
    for chunk_method in chunk_methods:
        suffix = chunk_method.split(".")[0]
        logging.info(f"{suffix}-namespace-chunking-benchmark")

        if (args.run_indexing or args.run_reranking) and (not args.run_eval):
            # without eval: skip token_limit loop when run indexing or run reranking, no eval
            run_single_benchmark(
                text="...",
                chunking_method=ChunkingMethod.CUSTOM,
                vector_db=VectorDB.PINECONE,
                embedding_model=OpenAIEmbedder,
                metrics=[context_precision, context_recall],
                load_chunk_from_file=f"{chunk_dir}/{chunk_method}",
                namespace=f"{suffix}-namespace-chunking-benchmark",
                index_name=index_name,
                path_questions=args.path_questions,
                num_chunks=None,
                retriever_topk=args.topk_retrieval,
                rerank_topk=args.topk_rerank,
                num_questions=args.num_questions,
                to_rerank=args.to_rerank,
                res_dir=out_dir,
                run_indexing=args.run_indexing,
                run_reranking=args.run_reranking,
                run_eval=args.run_eval,
                rerank_method=args.rerank_method,
            )
        else:
            # with eval
            for t_limit in args.token_limit:
                tasks.append(
                    run_single_benchmark_async(
                        text="...",
                        chunking_method=ChunkingMethod.CUSTOM,
                        embedding_model=OpenAIEmbedder,
                        vector_db=VectorDB.PINECONE,
                        metrics=[context_precision, context_recall],
                        load_chunk_from_file=f"{chunk_dir}/{chunk_method}",
                        namespace=f"{suffix}-namespace-chunking-benchmark",
                        index_name=index_name,
                        retriever_topk=args.topk_retrieval,
                        rerank_topk=args.topk_rerank,
                        token_limit=t_limit,
                        num_questions=args.num_questions,
                        path_questions=args.path_questions,
                        num_chunks=None,
                        to_rerank=args.to_rerank,
                        res_dir=out_dir,
                        run_indexing=args.run_indexing,
                        run_reranking=args.run_reranking,
                        run_eval=args.run_eval,
                        rerank_method=args.rerank_method,
                    )
                )
                suffixes.append(f"{suffix}-tokenlimit-{t_limit}")
    results_async = await asyncio.gather(*tasks)
    for result, suffix in zip(results_async, suffixes):
        res[suffix] = result

    with open(f"{out_dir}/results.json", "w") as f:
        res = {k: v._repr_dict if not isinstance(v, dict) else v for k, v in res.items()}
        json.dump(res, f, indent=4)
    for k, v in res.items():
        logging.info(f"{k}: {v}")

    analyse_results(res, res_dir=out_dir)


if __name__ == "__main__":
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Chunking benchmark")
    parser.add_argument(
        "--chunk_dir",
        type=str,
        default="assets/corpora_public/chunks",
        help="Path to the directory containing the chunk files",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default="openai-index",
        help="Index name for the vector database, default openai-index",
    )
    parser.add_argument(
        "--path_questions",
        type=str,
        default="assets/corpora_public/questions.csv",
        help="Path to the .csv questions",
    )
    parser.add_argument(
        "--topk_retrieval",
        type=int,
        default=400,
        help="Top k chunks to retrieve from vector database, default 400",
    )
    parser.add_argument(
        "--topk_rerank",
        type=int,
        default=100,
        help="Top k chunks to rerank, default 100",
    )
    parser.add_argument(
        "--token_limit",
        nargs="+",
        type=int,
        help="Token limit for the context retrieval, default=[50, 150, 250]",
        default=[50, 150, 250],
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=None,
        help="Number of questions to evaluate, default all.",
    )
    parser.add_argument(
        "--to_rerank",
        action="store_true",
        help="Whether to use reranking during retrieval",
    )
    parser.add_argument(
        "--run_indexing",
        action="store_true",
        help="Run indexing",
    )
    parser.add_argument(
        "--run_reranking",
        action="store_true",
        help="Run retrieval",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Run evaluation",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out",
        help="Output directory for the results",
    )
    parser.add_argument(
        "--rerank_method",
        type=str,
        default="aws",
        help="Rerank method",
    )
    args = parser.parse_args()
    print(args)

    asyncio.run(main(args))
