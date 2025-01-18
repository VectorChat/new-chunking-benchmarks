# This chunker generates chunks chapter-by-chapter.
import argparse
from tqdm import tqdm
from glob import glob
import asyncio
import json
import os
from dotenv import load_dotenv
from chunking_benchmark.chunkers.ai21 import AI21ChunkRunner
from chunking_benchmark.chunkers.base import BaseChunkRunner
from chunking_benchmark.chunkers.miner import MinerChunkRunner
from chunking_benchmark.chunkers.subnet_api import SubnetAPIChunker
from chunking_benchmark.chunkers.unstructured import UnstructuredChunkerRunner
from chunking_benchmark.utils.chunks import get_chunk_stats, print_chunk_stats, save_chunk_size_hist_from_json
from tabulate import tabulate
from datetime import datetime

argparser = argparse.ArgumentParser()

argparser.add_argument("--vali_wallet", type=str, help="Coldkey for validator wallet to query individual miners")
argparser.add_argument("--vali_hotkey", type=str, help="Hotkey for validator wallet to query individual miners")
argparser.add_argument("--netuid", type=int, default=40)
argparser.add_argument("--chunk_size", type=int, default=400)
argparser.add_argument("--chunk_size_char_multiplier", type=int, default=4)
argparser.add_argument("--text_file", type=str, required=True, help="Path to text file or directory")
argparser.add_argument("--miner_uids", type=str, help="Comma separated list of miner uids to query")
argparser.add_argument("--min_chunk_size", type=int, default=28)
argparser.add_argument("--out_dir", type=str, default="out")
argparser.add_argument("--miner_timeout", type=int, default=120)


async def main():
    load_dotenv(override=True)

    args = argparser.parse_args()

    text_files = glob(os.path.join(args.text_file, "*.txt")) + glob(os.path.join(args.text_file, "*.md")) if os.path.isdir(args.text_file) else [args.text_file]
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    res, metas = {}, {}
    print(f"subnet_api_{os.getenv('SUBNET_API_METHOD', '')}")
    for text_file in tqdm(text_files):
        print(f"Processing {text_file}")
        chapter = os.path.basename(text_file).split(".")[0]
        chunk_runners: list[BaseChunkRunner] = [
            # UnstructuredChunkerRunner(chunk_size=args.chunk_size, chunk_size_char_multiplier=args.chunk_size_char_multiplier, text_file=text_file),
            # SubnetAPIChunker(
            #     chunker_name=f"{args.chunk_size}1ktoken_subnet_api_v0",
            #     max_chunk_size_chars=args.chunk_size,  # increase this to reduce num chunk
            #     max_num_chunks=1000,
            # ),
            # SubnetAPIChunker(
            #     chunker_name=f"{args.chunk_size}1ktoken_subnet_api_v1",
            #     max_chunk_size_chars=args.chunk_size,  # increase this to reduce num chunk
            #     max_num_chunks=1000,
            # ),
            # SubnetAPIChunker(
            #     chunker_name=f"{args.chunk_size}1ktoken_subnet_api_v2",
            #     max_chunk_size_chars=args.chunk_size,  # increase this to reduce num chunk
            #     max_num_chunks=1000,
            # ),
            # SubnetAPIChunker(
            #     chunker_name=f"{args.chunk_size}1ktoken_subnet_api_v3",
            #     max_chunk_size_chars=args.chunk_size,  # increase this to reduce num chunk
            #     max_num_chunks=1000,
            # ),

            AI21ChunkRunner(chunk_size=args.chunk_size, min_chunk_size=args.min_chunk_size),
            # SubnetAPIChunker(
            #     chunker_name=f"4000Nonetoken_subnet_api_{os.getenv('SUBNET_API_METHOD', '')}",
            #     max_chunk_size_chars=4000,
            #     max_num_chunks=None,
            # ),
            # SubnetAPIChunker(
            #     chunker_name=f"{args.chunk_size}1ktoken_subnet_api_{os.getenv('SUBNET_API_METHOD', '')}",
            #     max_chunk_size_chars=args.chunk_size,  # increase this to reduce num chunk
            #     max_num_chunks=1000,
            # ),
            # SubnetAPIChunker(
            #     chunker_name=f"2000Nonetoken_subnet_api_{os.getenv('SUBNET_API_METHOD', '')}",
            #     max_chunk_size_chars=2000,
            #     max_num_chunks=None,
            # ),
            # SubnetAPIChunker(
            #     chunker_name=f"token_subnet_api_{os.getenv('SUBNET_API_METHOD', '')}",
            #     max_chunk_size_chars=1200,
            #     max_num_chunks=None,
            # ),
        ]

        if args.miner_uids:
            miner_uids = map(int, args.miner_uids.split(","))
            for miner_uid in miner_uids:
                chunk_runners.append(
                    MinerChunkRunner(
                        uid=miner_uid,
                        chunk_size=args.chunk_size,
                        vali_wallet=args.vali_wallet,
                        vali_hotkey=args.vali_hotkey,
                        timeout=args.miner_timeout,
                        chunk_size_char_multiplier=args.chunk_size_char_multiplier,
                    )
                )

        with open(text_file, "r") as f:
            text = f.read()

        table_rows = []

        for chunk_runner in chunk_runners:
            print("-" * 80)
            print(f"Running {chunk_runner.name}")
            chunks, metadata = await chunk_runner.run(text)
            print(f"{chunk_runner.name}: made {len(chunks)} chunks")
            print(f"{chunk_runner.name} metadata: {json.dumps(metadata, indent=2)}")

            chunk_stats = get_chunk_stats(chunks)

            table_rows.append(
                [
                    chunk_runner.name,
                    chunk_stats["num_chunks"],
                    chunk_stats["avg_chunk_size"],
                    chunk_stats["max_chunk_size"],
                    chunk_stats["min_chunk_size"],
                    chunk_stats["median"],
                    chunk_stats["variance"],
                    chunk_stats["std_dev"],
                ]
            )

            out_path = os.path.join(args.out_dir, date_str, chunk_runner.name, f"{chapter}.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            chapter_chunk = [{"text": chunk, "chapter_name": chapter} for chunk in chunks]
            with open(out_path, "w") as f:
                json.dump(chapter_chunk, f, indent=4)
                print(f"Wrote {len(chunks)} chunks to {out_path}")
            res[chunk_runner.name] = (
                chapter_chunk if chunk_runner.name not in res else res[chunk_runner.name] + chapter_chunk
            )
            metas[chunk_runner.name] = [metadata] if chunk_runner.name not in metas else metas[chunk_runner.name] + [metadata]
            print(
                tabulate(
                    table_rows,
                    headers=[
                        "Chunker",
                        "Num Chunks",
                        "Avg Chunk Size",
                        "Max Chunk Size",
                        "Min Chunk Size",
                        "Median",
                        "Variance",
                        "Std Dev",
                    ],
                    tablefmt="rounded_grid",
                )
            )

    # Agreegate chunks from all chapters
    for k, v in res.items():
        metadata = get_chunk_stats([item["text"] for item in v])
        metadata["metadata_additional"] = metas.get(k, {})
        avg_chunk_size = int(metadata["avg_chunk_size"])
        out_path = os.path.join(args.out_dir, date_str, f"{k}-{avg_chunk_size}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f, open(out_path.replace(".json", "_metadata.json"), "w") as f_meta:
            json.dump(v, f, indent=4)
            json.dump(metadata, f_meta, indent=4)
            print(f"Wrote {len(v)} chunks to {out_path}")
        save_chunk_size_hist_from_json(out_path)
        print("Exported histogram to", out_path.replace(".json", ".png"))


if __name__ == "__main__":
    asyncio.run(main())
