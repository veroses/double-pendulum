import numpy as np


def rotation_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to the continuous 6D representation (Zhou et al. 2019).

    The 6D representation is simply the first two columns of the rotation matrix,
    stacked into a single vector. It is continuous (unlike quaternions or Euler
    angles), which makes it a better regression target for neural networks.

    Args:
        R: Rotation matrix/matrices of shape (..., 3, 3). The leading dimensions
           are treated as a batch and preserved.

    Returns:
        np.ndarray of shape (..., 6): [c0 (3,), c1 (3,)] where c0, c1 are the first
        two columns of R.
    """
    R = np.asarray(R, dtype=np.float64)
    c0 = R[..., :, 0]
    c1 = R[..., :, 1]
    return np.concatenate([c0, c1], axis=-1)


def rotation_from_6d(d6: np.ndarray) -> np.ndarray:
    """Reconstruct rotation matrices from the 6D representation via Gram-Schmidt.

    Inverts rotation_to_6d(). Given the (possibly non-orthonormal) first two column
    vectors, this orthonormalises them and recovers the third by a cross product,
    yielding a valid rotation in SO(3). Because it always projects back onto SO(3),
    it is safe to call on raw network outputs.

    Args:
        d6: Array of shape (..., 6): the two stacked column vectors [a0 (3,), a1 (3,)].
            The leading dimensions are treated as a batch and preserved.

    Returns:
        np.ndarray of shape (..., 3, 3): orthonormal rotation matrices with det +1.
    """
    d6 = np.asarray(d6, dtype=np.float64)
    a0 = d6[..., 0:3]
    a1 = d6[..., 3:6]

    eps = 1e-8
    b0 = a0 / np.clip(np.linalg.norm(a0, axis=-1, keepdims=True), eps, None)
    # Remove the b0 component from a1, then normalise.
    proj = np.sum(b0 * a1, axis=-1, keepdims=True) * b0
    b1 = a1 - proj
    b1 = b1 / np.clip(np.linalg.norm(b1, axis=-1, keepdims=True), eps, None)
    b2 = np.cross(b0, b1)

    # Stack the three orthonormal vectors as columns.
    return np.stack([b0, b1, b2], axis=-1)
