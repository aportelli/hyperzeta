from typing import Optional

import mlx.core as mx
import numpy as np


class Lattice:
    """Helper class for linear lattice."""

    _n_min: int = -1
    _n_max: int = -1
    _nd: int = -1
    _dtype: mx.Dtype
    _no_zero: bool
    _basis: Optional[mx.array] = None
    _covolume: float = 1.0
    _gram: mx.array
    _n: mx.array
    _n_norm: mx.array
    _n_hat: mx.array

    @property
    def nd(self) -> int:
        """Lattice number of dimensions."""
        return self._nd

    @nd.setter
    def nd(self, new_nd: int) -> None:
        if self.nd != new_nd:
            self._nd = new_nd
            self._make_lattice()

    @property
    def size(self) -> int:
        return self.n.shape[0]

    @property
    def n_min(self) -> int:
        """Minimum lattice component magnitude in the lattice."""
        return self._n_min

    @n_min.setter
    def n_min(self, new_n_min: int) -> None:
        if new_n_min != self.n_min:
            self._n_min = new_n_min
            self._make_lattice()

    @property
    def n_max(self) -> int:
        """Maximum lattice component magnitude in the lattice."""
        return self._n_max

    @n_max.setter
    def n_max(self, new_n_max: int) -> None:
        if new_n_max != self.n_max:
            self._n_max = new_n_max
            self._make_lattice()

    @property
    def basis(self) -> mx.array:
        if self._basis is not None:
            return self._basis
        else:
            return mx.eye(self.nd, dtype=self.dtype)

    @property
    def gram(self) -> mx.array:
        return self._gram

    @basis.setter
    def basis(self, new_basis: mx.array) -> None:
        self._basis = new_basis.astype(self.dtype)
        self._refresh_quantities()
        self._make_lattice()

    @property
    def covolume(self) -> float:
        return self._covolume

    @property
    def dtype(self) -> mx.Dtype:
        """dtype used for cached lattice arrays."""
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: mx.Dtype) -> None:
        if new_dtype != self.dtype:
            self._dtype = new_dtype
            self._make_lattice()

    @property
    def n(self) -> mx.array:
        """Lattice points."""
        return self._n

    @property
    def n_norm(self) -> mx.array:
        """Norms of the lattice points."""
        return self._n_norm

    @property
    def n_hat(self) -> mx.array:
        """Directions of the lattice points."""
        return self._n_hat

    def __init__(
        self,
        *,
        n_min: int = 0,
        n_max: int = 0,
        no_zero: bool = False,
        basis: Optional[mx.array] = None,
        nd: int = 3,
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        """
        Lattice of vectors n_1 * b_1 + ... + n_d * b_d with n in Z^d and b_i a basis of R^d.

        - `nd`: number of dimensions d
        - `n_min`: minimum value of the coordinates n_i
        - `n_max`: maximum value of the coordinates n_i
        - `no_zero`: exclude the zero size (i.e. n = [0, ..., 0])
        - `basis`: basis (d,d) matrix, column i is the basis vector b_i
        - `dtype`: MLX dtype lattice points
        """
        self._nd = nd
        self._n_min = n_min
        self._n_max = n_max
        self._dtype = dtype
        self._no_zero = no_zero
        if basis is not None:
            self._basis = basis.astype(self._dtype)
        self._refresh_quantities()
        self._make_lattice()

    def _refresh_quantities(self) -> None:
        if self._basis is not None:
            self._covolume = float(
                np.linalg.det(np.array(self._basis, dtype=np.float64))
            )
            self._gram = self.basis.T @ self.basis
        else:
            self._covolume = 1.0
            self._gram = mx.eye(self.nd, dtype=self.dtype)

    def _make_lattice(self) -> None:
        """Rebuild lattice points, norms, and directions."""
        if self.n_min < 0:
            raise ValueError(f"n_min must be larger than 0 (got {self.n_min})")
        if self.n_min > self.n_max:
            raise ValueError(
                f"n_min must be smaller than n_max (got {self.n_min} and {self.n_max})"
            )
        if self._basis is not None:
            if (
                self._basis.ndim != 2
                or self._basis.shape[0] != self.nd
                or self._basis.shape[1] != self.nd
            ):
                raise ValueError(
                    f"provided basis matrix is not a nd x nd matrix (got {self._basis.shape}, nd={self.nd})"
                )

        if self.n_min == 0:
            a = mx.arange(-self.n_max, self.n_max + 1)
        else:
            neg = mx.arange(-self.n_max, -self.n_min + 1)
            pos = mx.arange(self.n_min, self.n_max + 1)
            a = mx.concatenate([neg, pos], axis=0)
        xj = mx.meshgrid(*([a] * self.nd), indexing="ij")
        buf = mx.stack([*xj], axis=-1).reshape(-1, self.nd)
        if self._no_zero and self.n_min == 0:
            mid = buf.shape[0] // 2
            self._n = mx.concatenate([buf[:mid], buf[mid + 1 :]], axis=0).astype(
                self.dtype
            )
        else:
            self._n = buf.astype(self.dtype)
        if self._basis is not None:
            self._n = self._n @ self._basis.T
        self._n_norm = mx.linalg.norm(self._n, axis=1)
        self._n_hat = mx.where(
            self._n_norm.reshape(-1, 1) == 0,
            mx.zeros_like(self._n),
            self._n / self._n_norm.reshape(-1, 1),
        )
