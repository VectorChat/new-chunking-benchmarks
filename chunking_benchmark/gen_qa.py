import os
import logging
import argparse
import pandas as pd
import json
from datetime import datetime

from data_gen.synthetic_evaluation import SyntheticEvaluation
from data_gen.similarity_tuner import SimilarityTuner
from data_gen.download_public_corpora import download_openwebtext, download_wikitext
from data_gen.download_public_corpora import PUBLIC_DATASET, OUTPUT_DIR, MAPPING

logging.basicConfig(level=getattr(logging, os.getenv("LOGLEVEL", "INFO").upper()))


def gen_qa(
    corpora_paths,
    queries_csv_path,
    queries_per_corpus,
    approx_excerpts,
    thres_duplicate,
    thres_poor_excerpt,
    tune_duplicate_quantile,
    tune_poor_excerpt_quantile,
    **kwargs,
):
    logging.info("Init q&a pairs.")
    evaluation = SyntheticEvaluation(corpora_paths, queries_csv_path, openai_api_key=os.getenv("OPENAI_API_KEY"))
    evaluation.generate_queries_and_excerpts(
        approximate_excerpts=approx_excerpts, num_rounds=1, queries_per_corpus=queries_per_corpus
    )

    logging.info("Tuning similarity thresholds.")
    if thres_duplicate is None or thres_poor_excerpt is None:
        df = pd.read_csv(queries_csv_path)
        df["references"] = df.references.apply(json.loads)
        queries = df.question.tolist()
        excerpts = df.references.apply(lambda x: [_.get("content", "") for _ in x]).tolist()
        tuner = SimilarityTuner()
        if thres_duplicate is None:
            thres_duplicate, _ = tuner.tune_all_query_pairs(queries, queries, target_quantile=tune_duplicate_quantile)
        if thres_poor_excerpt is None:
            thres_poor_excerpt, _ = tuner.tune_query_vs_excepts(
                queries, excerpts, target_quantile=tune_poor_excerpt_quantile
            )

    logging.info("Apply the tuned thresholds to filter the generated pairs.")
    evaluation.filter_poor_excerpts(threshold=thres_poor_excerpt)
    evaluation.filter_duplicates(threshold=thres_duplicate)
    logging.debug("Gererated Q&A pairs at: {queries_csv_path}")


def download_corpora(args):
    out_paths = []
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"{args.output_dir}/{date_str}"
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    logging.info(f"Downloading dataset to: {args.output_dir}")

    for dataset_name in args.dataset_name:
        logging.info(f"Downloading {dataset_name}")
        func = MAPPING.get(dataset_name, None)
        if MAPPING.get(dataset_name, None):
            out_paths.extend(func(args))
        else:
            logging.warning(f"Dataset {dataset_name} not found in MAPPING.")
    return out_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation dataset.")
    # download public corpora
    parser.add_argument("--dataset_name", nargs="+", type=str, default=PUBLIC_DATASET)
    parser.add_argument("--min_len", type=int, default=75000, help="Min length in characters.")
    parser.add_argument("--max_len", type=int, default=1000000, help="Max length in characters.")
    parser.add_argument("--num_docs", type=int, default=3, help="Number of documents to download.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    # init q&a pairs
    parser.add_argument("--queries_csv_path", help="Path to output CSV file.", required=True, default="generated_queries_excerpts.csv")  # fmt: skip
    parser.add_argument("--queries_per_corpus", type=int, default=2, help="Number of queries per corpus.")
    parser.add_argument("--approx_excerpts", action="store_true", help="Generate approximate excerpts.")
    # tune and apply similarity thresholds
    parser.add_argument("--thres_poor_excerpt", type=float, default=None, help="Threshold to filter poor excerpts, default using tuner.")  # fmt: skip
    parser.add_argument("--thres_duplicate", type=float, default=None, help="Threshold to filter duplicate excerpts, default using tuner.")  # fmt: skip
    parser.add_argument("--tune_duplicate_quantile", type=float, default=0.1, help="Target quantile for tuning duplicate.")  # fmt: skip
    parser.add_argument("--tune_poor_excerpt_quantile", type=float, default=0.1, help="Target quantile for tuning poor excerpt.")  # fmt: skip

    args = parser.parse_args()

    logging.info(f"Generating Q&A following arguments: {args}")
    corpora_paths = download_corpora(args)
    [logging.info(f"Downloaded corpora to: {_}") for _ in corpora_paths]

    gen_qa(
        corpora_paths,
        args.queries_csv_path,
        args.queries_per_corpus,
        args.approx_excerpts,
        args.thres_duplicate,
        args.thres_poor_excerpt,
        args.tune_duplicate_quantile,
        args.tune_poor_excerpt_quantile,
    )
