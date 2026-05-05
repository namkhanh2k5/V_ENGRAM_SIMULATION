from typing import Any, Optional, Tuple

import numpy as np


def normalize_L2(x: np.ndarray) -> None: ...

def vector_to_array(x: Any) -> np.ndarray: ...


class _IndexBase:
    ntotal: int

    def add(self, x: np.ndarray, numeric_type: int = 0) -> None: ...

    def search(
        self,
        x: np.ndarray,
        k: int,
        *,
        params: Any | None = None,
        D: Optional[np.ndarray] = None,
        I: Optional[np.ndarray] = None,
        numeric_type: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]: ...


class IndexFlatIP(_IndexBase):
    def __init__(self, d: int) -> None: ...


class IndexFlatL2(_IndexBase):
    def __init__(self, d: int) -> None: ...


class _PQInner:
    centroids: Any


class IndexPQ(_IndexBase):
    pq: _PQInner

    def __init__(self, d: int, m: int, nbits: int) -> None: ...

    def train(self, x: np.ndarray) -> None: ...

    def sa_encode(self, x: np.ndarray, codes: Any | None = None) -> np.ndarray: ...
