import mlx.core as mx


class Grid:
    _n_min: int = -1
    _n_max: int = -1
    _nd: int = -1
    _no_zero: bool
    _grid: mx.array
    _n_norm: mx.array
    _n_hat: mx.array

    @property
    def nd(self) -> int:
        return self._nd

    @nd.setter
    def nd(self, new_nd: int) -> None:
        if self.nd != new_nd:
            self._nd = new_nd
            self._make_grid()

    @property
    def n_min(self) -> int:
        return self._n_min

    @n_min.setter
    def n_min(self, new_n_min: int) -> None:
        if new_n_min != self.n_min:
            self._n_min = new_n_min
            self._make_grid()

    @property
    def n_max(self) -> int:
        return self._n_max

    @n_max.setter
    def n_max(self, new_n_max: int) -> None:
        if new_n_max != self.n_max:
            self._n_max = new_n_max
            self._make_grid()

    @property
    def grid(self) -> mx.array:
        return self._grid

    @property
    def n_norm(self) -> mx.array:
        return self._n_norm

    @property
    def n_hat(self) -> mx.array:
        return self._n_hat

    def __init__(
        self, *, n_min: int = 0, n_max: int = 0, no_zero: bool = False, nd: int = 3
    ) -> None:
        self._nd = nd
        self._n_min = n_min
        self._n_max = n_max
        self._no_zero = no_zero
        self._make_grid()

    def _make_grid(self) -> None:
        if self.n_min < 0:
            raise ValueError(f"n_min must be larger than 0 (got {self.n_min})")
        if self.n_min > self.n_max:
            raise ValueError(
                f"n_min must be smaller than n_max (got {self.n_min} and {self.n_max})"
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
            self._grid = mx.concatenate([buf[:mid], buf[mid + 1 :]], axis=0)
        else:
            self._grid = buf
        self._n_norm = mx.linalg.norm(self._grid, axis=1)
        self._n_hat = self._grid / self._n_norm.reshape(-1, 1)
