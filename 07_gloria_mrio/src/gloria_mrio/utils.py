"""Utility helpers; matrix inversion via scipy.linalg.lu_factor + lu_solve."""

import numpy as np
from tqdm import tqdm


def mem_gb() -> float:
    """Get current memory usage in GB"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**3
    except:
        return float("nan")


def inverse_with_progress(B: np.ndarray, dtype=np.float32,
                          block_size: int = 2000, desc: str = "Solving") -> np.ndarray:
    """
    Matrix inversion using scipy LU factorization (much faster).

    LU factorize once, then solve in blocks for memory efficiency.
    This avoids redundant factorization per block in the original version.

    Args:
        B: Input matrix
        dtype: Output data type
        block_size: Block size for solving (memory management)
        desc: Progress bar description

    Returns:
        Inverse of B
    """
    n = B.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=dtype)

    try:
        from scipy.linalg import lu_factor, lu_solve
        print(f"    Using scipy LU factorization (n={n})...")

        # LU factorize ONCE (O(n^3) but only once, not per block)
        lu, piv = lu_factor(B.astype(np.float64))

        I = np.eye(n, dtype=np.float64)
        X = np.empty((n, n), dtype=dtype)

        n_blocks = (n + block_size - 1) // block_size
        for start in tqdm(range(0, n, block_size), total=n_blocks, desc=desc):
            end = min(start + block_size, n)
            X[:, start:end] = lu_solve((lu, piv), I[:, start:end]).astype(dtype)

        return X

    except ImportError:
        print(f"    scipy not available, falling back to numpy...")
        # Fallback to original block-wise numpy solve
        I = np.eye(n, dtype=dtype)
        X = np.empty((n, n), dtype=dtype)

        n_blocks = (n + block_size - 1) // block_size
        for start in tqdm(range(0, n, block_size), total=n_blocks, desc=desc):
            end = min(start + block_size, n)
            X[:, start:end] = np.linalg.solve(B, I[:, start:end])

        return X
