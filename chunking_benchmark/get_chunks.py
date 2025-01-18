import argparse
import asyncio
import json
import os
from dotenv import load_dotenv
from chunking_benchmark.chunkers.ai21 import AI21ChunkRunner
from chunking_benchmark.chunkers.base import BaseChunkRunner
from chunking_benchmark.chunkers.miner import MinerChunkRunner
from chunking_benchmark.chunkers.unstructured import UnstructuredChunkerRunner
from chunking_benchmark.utils.chunks import get_chunk_stats, print_chunk_stats
from tabulate import tabulate
from datetime import datetime

argparser = argparse.ArgumentParser()

argparser.add_argument("--vali_wallet", type=str, help="Coldkey for validator wallet to query individual miners")
argparser.add_argument("--vali_hotkey", type=str, help="Hotkey for validator wallet to query individual miners")
argparser.add_argument("--netuid", type=int, default=40)
argparser.add_argument("--chunk_size", type=int, default=400)
argparser.add_argument("--chunk_size_char_multiplier", type=int, default=4)
argparser.add_argument("--text_file", type=str, required=True)
argparser.add_argument("--miner_uids", type=str, help="Comma separated list of miner uids to query")
argparser.add_argument("--min_chunk_size", type=int, default=28)
argparser.add_argument("--out_dir", type=str, default="out")
argparser.add_argument("--miner_timeout", type=int, default=120)

async def main():
    load_dotenv(override=True)
    
    args = argparser.parse_args()
    
    chunk_runners: list[BaseChunkRunner] = [
        UnstructuredChunkerRunner(chunk_size=args.chunk_size, chunk_size_char_multiplier=args.chunk_size_char_multiplier, text_file=args.text_file),        
        AI21ChunkRunner(chunk_size=args.chunk_size, min_chunk_size=args.min_chunk_size)
    ]
    
    if args.miner_uids:
        miner_uids = map(int, args.miner_uids.split(","))
        for miner_uid in miner_uids:
            chunk_runners.append(
                MinerChunkRunner(uid=miner_uid, chunk_size=args.chunk_size, vali_wallet=args.vali_wallet, vali_hotkey=args.vali_hotkey, timeout=args.miner_timeout, chunk_size_char_multiplier=args.chunk_size_char_multiplier)
            )
    
    with open(args.text_file, "r") as f:
        text = f.read()
                    
    table_rows = []
          
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")   
                      
    for chunk_runner in chunk_runners:
        print("-"*80)
        print(f"Running {chunk_runner.name}")
        chunks, metadata = await chunk_runner.run(text)
        print(f"{chunk_runner.name}: made {len(chunks)} chunks")                        
        print(f"{chunk_runner.name} metadata: {json.dumps(metadata, indent=2)}")
        
        chunk_stats = get_chunk_stats(chunks)
        
        table_rows.append([
            chunk_runner.name,
            chunk_stats["num_chunks"],
            chunk_stats["avg_chunk_size"],
            chunk_stats["max_chunk_size"],
            chunk_stats["min_chunk_size"],
            chunk_stats["median"],
            chunk_stats["variance"],
            chunk_stats["std_dev"]
        ])                
        
        out_path = os.path.join(args.out_dir, date_str, f"{chunk_runner.name}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(chunks, f)
            print(f"Wrote {len(chunks)} chunks to {out_path}")
                
    print(tabulate(table_rows, headers=["Chunker", "Num Chunks", "Avg Chunk Size", "Max Chunk Size", "Min Chunk Size", "Median", "Variance", "Std Dev"], tablefmt="rounded_grid"))

    

if __name__ == "__main__":        
    asyncio.run(main())
