from typing import Any, Callable, Optional, Sequence
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


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
