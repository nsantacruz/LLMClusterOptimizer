from typing import Any, Callable, Optional, Sequence
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import silhouette_score
from sklearn.base import ClusterMixin
from numpy import ndarray


def get_by_xml_tag(text, tag_name) -> Optional[str]:
    match = re.search(fr'<{tag_name}>(.+?)</{tag_name}>', text, re.DOTALL)
    if not match:
        return None
    return match.group(1)


def run_parallel(items: Sequence[Any], unit_func: Callable, max_workers: int, **tqdm_kwargs) -> list:
    def _pbar_wrapper(pbar, item):
        unit = unit_func(item)
        with pbar.get_lock():
            pbar.update(1)
        return unit

    with tqdm(total=len(items), **tqdm_kwargs) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for item in items:
                futures.append(executor.submit(_pbar_wrapper, pbar, item))

    output = [future.result() for future in futures if future.result() is not None]
    return output


def guess_optimal_n_clusters(embeddings: ndarray, get_model: Callable[[int], ClusterMixin], verbose=False) -> int:
    """
    Utility function to guess optimal number of clusters for a given cluster model.
    Useful for models that require number of clusters as input.
    :param embeddings: ndarray of embeddings of shape (n_samples, n_features)
    :param get_model: Callable that takes number of clusters as parameter and returns an sklearn cluster model
    :param verbose:
    :return: optimal number of clusters based on silhouette score
    """
    if len(embeddings) <= 1:
        return len(embeddings)

    best_sil_coeff = -1
    best_num_clusters = 0
    MAX_MIN_CLUSTERS = 3  # the max start of the search for optimal cluster number.
    n_cluster_start = min(len(embeddings), MAX_MIN_CLUSTERS)
    n_cluster_end = len(embeddings)//2
    if n_cluster_end < (n_cluster_start + 1):
        n_cluster_start = 2
        n_cluster_end = n_cluster_start + 1
    n_clusters = range(n_cluster_start, n_cluster_end)
    for n_cluster in tqdm(n_clusters, total=len(n_clusters), desc='guess optimal clustering', disable=not verbose):
        labels = get_model(n_cluster).fit_predict(embeddings)
        sil_coeff = silhouette_score(embeddings, labels, metric='cosine')
        if sil_coeff > best_sil_coeff:
            best_sil_coeff = sil_coeff
            best_num_clusters = n_cluster
    if verbose:
        print("Best N", best_num_clusters, "Best silhouette score", round(best_sil_coeff, 4))
    return best_num_clusters
