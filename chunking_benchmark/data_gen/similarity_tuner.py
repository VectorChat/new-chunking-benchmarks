from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from tqdm import tqdm


class SimilarityTuner:
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        q_list: list = [0.1, 0.25, 0.5, 0.75, 0.9],
        **kwargs,
    ):
        self.client = OpenAI()
        self.embedding_model = embedding_model
        self.q_list = q_list

    def get_embedding(self, texts: list[str], **kwargs):
        response = self.client.embeddings.create(input=texts, model=self.embedding_model)
        return [embedding.embedding for embedding in response.data]

    def cal_similarity(self, first: list, second: list[str], **kwargs):
        embed_first = self.get_embedding(first)
        embed_second = self.get_embedding(second)
        similarity_matrix = cosine_similarity(embed_first, embed_second)
        similarities = similarity_matrix.flatten()
        similarities.sort()
        return similarities

    def quantiles(self, similarities: list[float], q_list: list = [0.1, 0.25, 0.5, 0.75, 0.9], **kwargs):
        return {q: similarities[int(len(similarities) * q)] for q in q_list}

    def hist(self, similarities: list[float], output_name: str = "tmp.png", **kwargs):
        plt.hist(similarities, bins=100)
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.savefig(output_name)

    def tune_all_query_pairs(self, first: list[str], second: list[str], target_quantile: float = 0.5, **kwargs):
        """Tune the similarity threshold for all pairs of questions."""
        similarities = self.cal_similarity(first, second)
        self.hist(similarities, output_name="tune_all_query_pairs.png")
        quantiles = self.quantiles(similarities, self.q_list)
        return quantiles.get(target_quantile, None), quantiles

    def tune_query_vs_excepts(self, queries: list, excepts_s: list[list], target_quantile: float = 0.5, **kwargs):
        """Tune the similarity threshold for all (query vs. corresponding excepts)."""
        assert len(queries) == len(excepts_s)
        similarities = []
        for query, excepts in tqdm(zip(queries, excepts_s), total=len(queries), desc="Tuning query vs. excepts"):
            sims = self.cal_similarity([query], excepts)
            similarities.extend(sims)

        self.hist(similarities, output_name="tune_query_vs_excepts.png")
        quantiles = self.quantiles(similarities, self.q_list)
        return quantiles.get(target_quantile, None), quantiles


if __name__ == "__main__":
    tuner = SimilarityTuner()

    threshold, quantiles = tuner.tune_all_query_pairs(
        ["Hello, world!", "hi there"], ["Hello, world!", "Thank you", "cheers"]
    )
    print(threshold, quantiles)

    threshold, quantiles = tuner.tune_query_vs_excepts(
        ["Hello, world!", "hi there"],
        [["Hello, world!", "Thank you", "cheers"], ["Hello, world!", "Thank you", "cheers", "Yeah"]],
    )
    print(threshold, quantiles)
