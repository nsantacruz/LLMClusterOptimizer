# Project Description

## llm-cluster-optimizer

A drop-in replacement for sklearn clustering models, this package optimizes sklearn clusters using LLMs

## Installation

By default, this package relies on OpenAI for LLM interactions. This requires having copying your API key from [here](https://platform.openai.com/account/api-keys) and putting it in the `OPENAI_API_KEY` envvar.

Otherwise, you can use any other LLM provider, as described below.

To install `llm-cluster-optimizer`, run the following command in your terminal:

```shell
pip install llm-cluster-optimizer
```

## Usage

Here is a simple example of usage that provides two cluster models as options to be optimized. `labels` will be an `np.ndarray` of shape `(len(input_texts),)` representing which cluster each input text falls into. Use `verbose=True` in `LLMClusterOptimizer.__init__()` to get more information on progress.

```python
from llm_cluster_optimizer import LLMClusterOptimizer
from sklearn.cluster import AffinityPropagation, HDBSCAN
import numpy as np
from langchain_openai import OpenAIEmbeddings

def embed_text_openai(text):
    return np.array(OpenAIEmbeddings(model="text-embedding-3-large").embed_query(text))

options = [
    HDBSCAN(min_cluster_size=2, min_samples=1),
    AffinityPropagation(damping=0.7),
]
optimizer = LLMClusterOptimizer(options, embed_text_openai)

input_texts = []  # your list of input texts
labels = optimizer.fit_predict_text(input_texts)
```

To use a clustering algorithm that takes the number of clusters as an input, you likely want to optimize `n` using the silhouette score or a similar metric. This requires access to the embeddings. You can pass the model wrapped in a function to achieve this:

```python
from sklearn.cluster import AgglomerativeClustering
from util import guess_optimal_n_clusters

def get_agglomerative_model(embeddings: np.ndarray):
    n_clusters = guess_optimal_n_clusters(embeddings, lambda n: AgglomerativeClustering(n_clusters=n))
    return AgglomerativeClustering(n_clusters=n_clusters)


options = [
    HDBSCAN(min_cluster_size=2, min_samples=1),
    AffinityPropagation(damping=0.7),
    get_agglomerative_model,
]
optimizer = LLMClusterOptimizer(options, embed_text_openai)
```

## Docs

### __init__()

| **Param**                               | **Description**                                                                                                                                                                                     | **Default**                                                                                            | **Type**                                                                  |
|-----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| `cluster_models`                        | List of SKlearn cluster model instances (`ClusterMixin`) to choose from. Each item can be either a `ClusterMixin` or a `Callable` that takes a list of embeddings and returns a `ClusterMixin`.     | N/A                                                                                                    | `Sequence[Union[ClusterMixin, Callable[[ndarray], ClusterMixin]]]`         |
| `embedding_fn`                          | Function that takes a string and returns its embedding.                                                                                                                                             | N/A                                                                                                    | `Callable[[str], ndarray]`                                                 |
| `recluster_model`                       | `ClusterMixin` used for reclustering results during optimization stages. Returns clusters that are too small. If not passed, `AffinityPropagation(damping=0.7)` is used.                            | `AffinityPropagation(damping=0.7)`                                                                     | `ClusterMixin`                                                             |
| `get_cluster_summary`                   | Function that takes a list of strings which are samples from a cluster and returns a summary of them. See `default` for default functionality. Requires OpenAI API key in envvars if using default. | Uses GPT-4o-mini to summarize strings sampled from the cluster. Summary will be no more than 10 words. | `Callable[[list[str]], str]`                                               |
| `calculate_clustering_score`            | Function that takes a list of `SummarizedCluster` and returns a float representing the clustering quality. Higher is better. See `default` for default functionality.                               | Calculates the average minimum pairwise cosine distance between cluster summary embeddings.            | `Callable[[list[SummarizedCluster]], float]`                              |
| `cosine_distance_thresholds_to_combine` | List of floats representing maximum cosine distance between cluster summary embeddings where clusters should be combined. The best threshold is chosen based on `calculate_clustering_score()`.     | N/A                                                                                                    | `Sequence[float]`                                                          |
| `summary_sample_size`                   | Number of items to sample from a cluster.                                                                                                                                                           | `5`                                                                                                    | `int`                                                                     |
| `num_summarization_workers`             | Number of workers to use when running `get_cluster_summary()`. Be aware of rate limits depending on the LLM provider being used.                                                                    | `25`                                                                                                   | `int`                                                                     |
| `num_embed_workers`                     | Number of workers to use when running `embedding_fn()`. Be aware of rate limits depending on the LLM provider being used.                                                                           | `50`                                                                                                   | `int`                                                                     |
| `verbose`                               | Output logs to console.                                                                                                                                                                             | `True`                                                                                                 | `bool`                                                                    |


### fit_predict_text()

Mimics `sklearn.base.ClusterMixin.fit_predict()`. Given a sequence of texts, returns the cluster labels for each text.
Chooses best of `cluster_models` passed in `__init__()` based on `calculate_clustering_score()`
Post-processes clusters so that:
- large clusters are broken up
- small clusters are combined based on similarity

| **Param**                               | **Description**                                                                                                                                                                                                                           | **Default**                                            | **Type**                                                                  |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------------|
| `texts` | List of strings to cluster | N/A | `list[str]` |