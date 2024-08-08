from llm_clusterer import LLMClusterOptimizer
from sklearn.cluster import AffinityPropagation, HDBSCAN, KMeans
import numpy as np
from langchain_openai import OpenAIEmbeddings
import json
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from util import guess_optimal_n_clusters
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

TEXT_DIR = "/Users/nss/sefaria/llm/experiments/topic_source_curation/_cache"

def get_texts_for_slug(slug):
    filename = f"{TEXT_DIR}/gathered_sources_{slug}.json"
    texts = []
    with open(filename, "r") as fin:
        jin = json.load(fin)
        for doc in jin:
            texts += [doc['source']["text"]['en']]
    return texts


def embed_text_openai(text):
    return np.array(OpenAIEmbeddings(model="text-embedding-3-large").embed_query(text))


def get_kmeans(embeddings):
    n_clusters = guess_optimal_n_clusters(embeddings, lambda n: KMeans(n_clusters=n))
    return KMeans(n_clusters=n_clusters)


def build_optimizer():
    recluster_model = AffinityPropagation(damping=0.7, max_iter=1000, convergence_iter=100)
    options = [
        HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_method="eom", cluster_selection_epsilon=0.65),
        HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_method="leaf", cluster_selection_epsilon=0.5),
        AffinityPropagation(damping=0.7, max_iter=1000, convergence_iter=100),
    ]
    return LLMClusterOptimizer(options, embed_text_openai, recluster_model, verbose=True, OPTIMIZE=True)


def run_example(slug):
    texts = get_texts_for_slug(slug)
    print("Num texts", len(texts))
    optimizer = build_optimizer()
    print("Optimizer", optimizer)
    labels, clusters = optimizer.fit_predict_text(texts)
    for cluster in clusters:
        print(f"({len(cluster)})", cluster.summary)


if __name__ == '__main__':
    run_example("redemption-of-captives")
