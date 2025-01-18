import logging
from datasets import load_dataset
import argparse
from tqdm import tqdm
import os
import json
from datetime import datetime
import random


PUBLIC_DATASET = ["openwebtext", "wikitext"]
OUTPUT_DIR = "assets/corpora_public"


def download_openwebtext(args):
    # https://huggingface.co/datasets/stas/openwebtext-10k
    dataset_name = "stas/openwebtext-10k"
    dataset_name_precise = dataset_name.split("/")[-1]
    out_paths = []

    ds = load_dataset(dataset_name, split="train")
    ds = ds.filter(lambda x: args.min_len < len(x["text"]) < args.max_len)
    print(f"Loaded {len(ds)} documents, len={[len(x['text']) for x in ds]}")
    ds = ds.select(range(min(args.num_docs, len(ds))))
    print(f"Selected {len(ds)} documents, len={[len(x['text']) for x in ds]}")
    for i, x in tqdm(enumerate(ds), total=len(ds)):
        out_path = f"{args.output_dir}/{dataset_name_precise}_{i}.txt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(x["text"])
            print(f"Written to {out_path}")
            out_paths.append(out_path)

    metadata = {**args.__dict__}
    metadata.update({"num_docs": len(ds)})
    with open(f"{args.output_dir}/metadata.json", "w") as f:
        f.write(json.dumps(metadata, indent=4))
        print(f"Written metadata to {args.output_dir}/metadata.json")
    return out_paths


def download_wikitext(args):
    # https://huggingface.co/datasets/Salesforce/wikitext
    # This dataset store raw text line by line, so we need to merge them into a big document,
    # then split to expected size (100k characters).
    dataset_name = "Salesforce/wikitext"
    dataset_name_precise = dataset_name.split("/")[-1]
    out_paths = []
    CHUNK_SIZE = 100000
    ds = load_dataset(dataset_name, "wikitext-103-raw-v1", split="train")
    jointed = "\n".join(ds["text"])
    print(f"Loaded {len(jointed)//CHUNK_SIZE} documents, {CHUNK_SIZE} characters each.")
    selected_id = random.sample([_ for _ in range(len(jointed)//CHUNK_SIZE)], args.num_docs)
    print(f"Selected {selected_id}")
    selected = [jointed[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] for i in selected_id]
    for i, x in tqdm(enumerate(selected), total=len(selected)):
        out_path = f"{args.output_dir}/{dataset_name_precise}_{selected_id[i]}.txt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(x)
            print(f"Written to {out_path}")
            out_paths.append(out_path)

    metadata = {**args.__dict__}
    metadata.update({"num_docs": len(selected)})
    with open(f"{args.output_dir}/metadata.json", "w") as f:
        f.write(json.dumps(metadata, indent=4))
        print(f"Written metadata to {args.output_dir}/metadata.json")
    return out_paths


def download_book_corpus(args):
    # bookcorpus/bookcorpus
    return None


MAPPING = {
    "openwebtext": download_openwebtext,
    "wikitext": download_wikitext,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", nargs="+", type=str, default=PUBLIC_DATASET)
    parser.add_argument("--min_len", type=int, default=75000)
    parser.add_argument("--max_len", type=int, default=1000000)
    parser.add_argument("--num_docs", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(args)

    for dataset_name in args.dataset_name:
        print(f"Downloading {dataset_name}")
        func = MAPPING.get(dataset_name, None)
        func(args) if func is not None else logging.warning(f"Dataset {dataset_name} not found in MAPPING.")
