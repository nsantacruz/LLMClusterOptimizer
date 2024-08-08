from typing import Sequence, Union, Callable
from functools import reduce, partial
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AffinityPropagation
from sklearn.base import ClusterMixin
import random
from dataclasses import dataclass
from numpy import ndarray
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from util import get_by_xml_tag, run_parallel
import numpy as np
from statistics import mean, stdev


@dataclass
class SummarizedCluster:
    label: int
    embeddings: list[ndarray]
    items: list[str]
    summary: str = None

    def merge(self, other: 'SummarizedCluster') -> 'SummarizedCluster':
        return SummarizedCluster(self.label, self.embeddings + other.embeddings, self.items + other.items, self.summary)

    def set_summary(self, summary: str) -> None:
        self.summary = summary.strip()

    @property
    def labels(self) -> list[int]:
        """
        list of `self.label` duplicated by how many items there are
        Useful for converting to sklearn `fit_predict()` return value
        :return:
        """
        return [self.label]*len(self)

    def __len__(self):
        return len(self.items)

    def __eq__(self, other):
        if not isinstance(other, SummarizedCluster):
            return False
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.label, len(self), self.summary))


def _default_get_cluster_summary(strs_to_summarize: Sequence[str]) -> str:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    system = SystemMessage(content="Given a few ideas (wrapped in <idea> "
                                   "XML tags) output a summary of the"
                                   "ideas. Wrap the output in <summary> tags. Summary"
                                   "should be no more than 10 words.")
    human = HumanMessage(content=f"<idea>{'</idea><idea>'.join(strs_to_summarize)}</idea>")
    response = llm.invoke([system, human])
    return get_by_xml_tag(response.content, "summary")


class LLMClusterOptimizer:

    def __init__(self, cluster_models: Sequence[Union[ClusterMixin, Callable[[ndarray], ClusterMixin]]],
                 embedding_fn: Callable[[str], ndarray], recluster_model: ClusterMixin = None,
                 get_cluster_summary: Callable[[list[str]], str] = None,
                 calculate_clustering_score: Callable[[list[SummarizedCluster]], float] = None,
                 cosine_distance_thresholds_to_combine: Sequence[float] = None,
                 summary_sample_size=5, num_summarization_workers=25, num_embed_workers=50, verbose=True):
        """
        :param cluster_models: list of SKlearn cluster model instances (aka ClusterMixins) to choose from. Each item can be either a ClusterMixin or a Callable that takes a list of embeddings and returns a ClusterMixin. Use the latter if the ClusterMixin requires n clusters as a parameter. The callable should use the embeddings to determine the optimal n. Use `util.guess_optimal_n_clusters()`.
        :param embedding_fn: Function that takes a strings and returns its embedding.
        :param recluster_model: ClusterMixin to be used for reclustering results during optimization stages. Ideally returns clusters that are too small. If not passed, AffinityPropagation(damping=0.7) will be used.
        :param get_cluster_summary: Functon that takes a list of strings which are samples from a cluster and returns a summary of them. If not passed `_default_get_cluster_summary()` will be used. If using default, make sure OpenAI API key is setup in envvars.
        :param calculate_clustering_score: Function that takes a list of `SummarizedCluster` and returns a float that represents how good the clustering is, higher is better. If not passed `self._default_calculate_clustering_score()` is used.
        :param cosine_distance_thresholds_to_combine: List of floats that represent maximum cosine distance between cluster summary embeddings where clusters should be combined. Best threshold will be chosen based on `calculate_clustering_score()`.
        :param summary_sample_size: Number of items to sample from a cluster
        :param num_summarization_workers: Number of workers to use when running `get_cluster_summary()`. Be aware of rate limits depending on the LLM provider being used.
        :param num_embed_workers: Number of workers to use when running `embedding_fn()`. Be aware of rate limits depending on the LLM provider being used.
        :param verbose: Output logs to console
        """
        self._cluster_models = cluster_models
        self._recluster_model = recluster_model or AffinityPropagation(damping=0.7)
        self._embedding_fn = embedding_fn
        self._get_cluster_summary = get_cluster_summary or _default_get_cluster_summary
        self._calculate_clustering_score = calculate_clustering_score or self._default_calculate_clustering_score
        self._cosine_distance_thresholds_to_combine = cosine_distance_thresholds_to_combine or [0.2, 0.25, 0.3]
        self._summary_sample_size = summary_sample_size
        self._num_summarization_workers = num_summarization_workers
        self._num_embed_workers = num_embed_workers
        self._verbose = verbose

    def fit_predict_text(self, texts: Sequence[str]) -> ndarray:
        """
        Mimics `sklearn.base.ClusterMixin.fit_predict()`. Given a sequence of texts, returns the cluster labels for each text.
        Chooses best of `cluster_models` passed in `__init__()` based on `calculate_clustering_score()`
        Post-processes clusters so that:
        - large clusters are broken up
        - small clusters are combined based on similarity
        :param texts: List of strings to cluster.
        :return: List of cluster labels of shape `(len(texts),)`
        """
        embeddings = self._embed_parallel(texts, desc="Embedding input texts")
        best_clusters = None
        highest_clustering_score = 0
        chosen_model = None
        for imodel, cluster_model in enumerate(self._cluster_models):
            if self._verbose:
                print("Clustering model: {}".format(cluster_model))
            curr_labels = self._get_labels(embeddings, cluster_model)
            curr_clusters = self._build_clusters(curr_labels, embeddings, texts)
            summarized_clusters = self._summarize_clusters(curr_clusters)
            summarized_clusters = self._optimize_collapse_similar_clusters(summarized_clusters)
            clustering_score = self._calculate_clustering_score(summarized_clusters)
            if clustering_score > highest_clustering_score:
                highest_clustering_score = clustering_score
                best_clusters = summarized_clusters
                chosen_model = self._cluster_models[imodel]
        if self._verbose:
            print("Highest clustering score", highest_clustering_score)
            print("Best model", chosen_model)
        return np.array(reduce(lambda x, y: x + y.labels, best_clusters, []))

    @staticmethod
    def _get_labels(embeddings: ndarray, cluster_model: Union[ClusterMixin, Callable[[ndarray], ClusterMixin]]) -> ndarray:
        if callable(cluster_model):
            cluster_model = cluster_model(embeddings)
        return cluster_model.fit_predict(embeddings)

    @staticmethod
    def _build_clusters_from_cluster_results(labels: ndarray, embeddings: ndarray, items: ndarray) -> (list[SummarizedCluster], list, ndarray):
        clusters = []
        noise_items = []
        noise_embeddings = []
        for label in np.unique(labels):
            indices = np.where(labels == label)[0]
            curr_embeddings = [embeddings[j] for j in indices]
            curr_items = [items[j] for j in indices]
            if label == -1:
                noise_items += curr_items
                noise_embeddings += curr_embeddings
                continue
            clusters += [SummarizedCluster(label, curr_embeddings, curr_items)]
        return clusters, noise_items, noise_embeddings

    def _build_clusters(self, labels, embeddings, texts) -> Sequence[SummarizedCluster]:
        clusters, noise_items, noise_embeddings = self._build_clusters_from_cluster_results(labels, embeddings, texts)
        clusters = self._recluster_large_clusters(clusters)
        if len(noise_items) > 0:
            noise_labels = self._recluster_model.fit(noise_embeddings).predict(noise_embeddings)
            noise_clusters, _, _ = self._build_clusters_from_cluster_results(noise_labels, noise_embeddings, noise_items)
            if self._verbose:
                print("LEN NOISE_CLUSTERS", len(noise_clusters))
            clusters += noise_clusters
        return clusters

    def _summarize_cluster(self, cluster: SummarizedCluster) -> SummarizedCluster:
        """
        :param cluster: Cluster to summarize
        :return: the same cluster object with the `summary` attribute set.
        """
        if len(cluster) == 1:
            summary = cluster.items[0]
        else:
            sample = random.sample(cluster.items, min(len(cluster), self._summary_sample_size))
            summary = self._get_cluster_summary(sample)
        cluster.set_summary(summary)
        return cluster

    def _summarize_clusters(self, clusters: Sequence[SummarizedCluster], **kwargs) -> list[SummarizedCluster]:
        return run_parallel(clusters, partial(self._summarize_cluster, **kwargs),
                            max_workers=self._num_summarization_workers, desc='summarize source clusters', disable=not self._verbose)

    def _embed_parallel(self, items: Sequence[str], **kwargs):
        return run_parallel(items, self._embedding_fn, max_workers=self._num_embed_workers, disable=not self._verbose, **kwargs)

    def _embed_cluster_summaries(self, summarized_clusters: Sequence[SummarizedCluster]):
        return self._embed_parallel([c.summary for c in summarized_clusters],
                                    desc="embedding cluster summaries to score")

    def _optimize_collapse_similar_clusters(self, clusters: Sequence[SummarizedCluster]) -> list[SummarizedCluster]:
        embeddings = self._embed_cluster_summaries(clusters)
        distances = pairwise_distances(embeddings, metric='cosine')
        highest_clustering_score = 0
        best_clusters = None
        for threshold in self._cosine_distance_thresholds_to_combine:
            temp_clusters = self._collapse_similar_clusters(clusters, distances, threshold)
            temp_score = self._calculate_clustering_score(temp_clusters)
            if temp_score > highest_clustering_score:
                highest_clustering_score = temp_score
                best_clusters = temp_clusters
        return best_clusters

    @staticmethod
    def _collapse_similar_clusters(clusters: Sequence[SummarizedCluster], distances: np.ndarray, threshold: float) -> list[SummarizedCluster]:
        merged = np.zeros(len(clusters), dtype=bool)
        new_clusters = []

        for i, curr_cluster in enumerate(clusters):
            if merged[i]:
                continue
            merged[i] = True
            # Find clusters that need to be merged with the current cluster
            to_merge = (distances[i] < threshold) & ~merged
            merged[to_merge] = True
            for j in np.where(to_merge)[0]:
                curr_cluster = curr_cluster.merge(clusters[j])
            new_clusters.append(curr_cluster)
        return new_clusters

    def _default_calculate_clustering_score(self, summarized_clusters: Sequence[SummarizedCluster]) -> float:
        """
        Default method to calculate score for how good clusters are, higher is better.
        Calculates the average minimum pairwise cosine distance between cluster summary embeddings.
        :param summarized_clusters:
        :return:
        """
        embeddings = self._embed_cluster_summaries(summarized_clusters)
        distances = pairwise_distances(embeddings, metric='cosine')
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        avg_min_distance = sum(min_distances)/len(min_distances)
        return avg_min_distance

    @staticmethod
    def _get_large_clusters(clusters: Sequence[SummarizedCluster]) -> list[SummarizedCluster]:
        large_clusters = []
        if len(clusters) <= 2:
            return large_clusters
        for cluster in clusters:
            other_cluster_lens = [len(c) for c in clusters if c != cluster]
            if len(cluster) > (mean(other_cluster_lens) + 3*stdev(other_cluster_lens)):
                large_clusters.append(cluster)
        return large_clusters

    def _recluster_large_clusters(self, clusters: Sequence[SummarizedCluster]) -> Sequence[SummarizedCluster]:
        large_clusters = set(self._get_large_clusters(clusters))
        other_clusters = [c for c in clusters if c not in large_clusters]
        if len(large_clusters) == 0:
            return clusters
        items_to_recluster = reduce(lambda x, y: x + y.items, large_clusters, [])
        embeddings = self._embed_parallel(items_to_recluster, desc="Reclustering large clusters")
        labels = self._recluster_model.fit_predict(embeddings)
        reclustered_clusters = self._build_clusters(labels, embeddings, items_to_recluster)
        return other_clusters + reclustered_clusters



