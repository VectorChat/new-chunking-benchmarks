import os
import numpy as np
import matplotlib.pyplot as plt
import json
import statistics

from chunking_benchmark.utils.tokens import get_num_tokens


def save_chunk_size_distribution(chunks: list[str], title: str, path: str):
    chunk_sizes = [get_num_tokens(chunk) for chunk in chunks]
    plt.hist(chunk_sizes, bins=100)
    plt.xlabel("Chunk Size")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.savefig(path)
    plt.clf()


def get_chunk_stats(chunks: list[str]):
    chunk_sizes = [get_num_tokens(chunk) for chunk in chunks]

    num_chunks = len(chunks)

    return {
        "avg_chunk_size": np.mean(chunk_sizes) if num_chunks > 0 else 0,
        "max_chunk_size": max(chunk_sizes) if num_chunks > 0 else 0,
        "min_chunk_size": min(chunk_sizes) if num_chunks > 0 else 0,
        "median": statistics.median(chunk_sizes) if num_chunks > 0 else 0,
        "variance": np.var(chunk_sizes) if num_chunks > 0 else 0,
        "std_dev": np.std(chunk_sizes) if num_chunks > 0 else 0,
        "num_chunks": num_chunks,
        "chunk_sizes": chunk_sizes,
    }


def print_chunk_stats(chunks: list[str]):
    chunk_sizes = [get_num_tokens(chunk) for chunk in chunks]

    avg_chunk_size = np.mean(chunk_sizes)
    print(f"avg chunk size: {avg_chunk_size}")
    print(f"max chunk size: {max(chunk_sizes)}")
    print(f"min chunk size: {min(chunk_sizes)}")
    print(f"median: {statistics.median(chunk_sizes)}")
    variance = np.var(chunk_sizes)
    print(f"variance: {variance}")
    std_dev = np.std(chunk_sizes)
    print(f"std dev: {std_dev}")
    print(f"num chunks: {len(chunks)}")


def stat_by_chapter(chunks, key_chapter_name: str = "chapter_name"):
    stats = {}
    for chunk in chunks:
        chapter, text = chunk.get("chapter_name", "unknown"), chunk.get("text", "")
        text_token = get_num_tokens(text)
        if chapter not in stats:
            stats[chapter] = [text_token]
        else:
            stats[chapter].append(text_token)
    stats = dict(sorted(stats.items(), reverse=True))
    return stats


def save_chunk_size_hist_from_json(path_data: str, by_chapter: bool = True, key_chapter_name: str = "chapter_name"):
    if not os.path.exists(path_data):
        return
    plt.figure(figsize=(6, 4))
    with open(path_data, "r") as f:
        chunks_with_metadata = json.load(f)
        if by_chapter:
            stats = stat_by_chapter(chunks_with_metadata, key_chapter_name)
            for chapter, values in stats.items():
                chapter_avg_token = int(np.mean(values))
                chapter_num_chunks = len(values)
                plt.hist(values, bins=100, alpha=0.75, label=f"{chapter} - {chapter_num_chunks} chunks - {chapter_avg_token} avg token", histtype="step")
        plt.hist([get_num_tokens(chunk["text"]) for chunk in chunks_with_metadata], bins=100, label="all", histtype="step")
    plt.xlabel("Chunk size (token)")
    plt.ylabel("Frequency")
    plt.title(os.path.basename(path_data) + f" - {len(chunks_with_metadata)} chunks")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    path_hist = path_data.replace(".json", ".png")
    plt.savefig(path_hist, dpi=120, bbox_inches="tight")
