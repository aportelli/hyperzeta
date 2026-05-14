import math
from typing import Optional

import mlx.core as mx
import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad

from hyperzeta.grid import Grid


def ak(k: float, v: float) -> float:
    if v < 0.0:
        raise RuntimeError(f"v must be positive (got {v})")
    if v == 0.0:
        return 1.0
    eps = k - 1.0
    if abs(eps) < 1e-8:
        return math.atanh(v) / v
    a = -eps * math.log1p(-v)
    b = -eps * math.log1p(v)
    return (math.expm1(a) - math.expm1(b)) / (2.0 * v * eps)


def _tanhsinh(r: float) -> float:
    if r > 40.0:
        return 1.0
    else:
        return math.tanh(math.sinh(r))


def _rj(j: float) -> float:
    def f(r: float) -> float:
        return (1.0 - _tanhsinh(r) ** (j + 2.0)) / (r ** (j - 2.0))

    result, _ = quad(f, 0.0, math.inf)
    return result


def _rbarj(j: float) -> float:
    def f(r: float) -> float:
        return _tanhsinh(r) ** (j + 2.0) / (r ** (j - 2.0))

    result, _ = quad(f, 0.0, math.inf)
    return result


class QedCoef:
    """
    Class computing the $c_j(v)$ QED finite-volume coefficients described in [1,2]

    References
    ----------
    [1]: [Davoudi et al., PRD 99(3), 114510 (2019)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.99.034510)
    [2]: [Di Carlo et al., PRD 105(7), 074509 (2022)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.074509)
    """

    _grid: Grid
    _last_j: Optional[float] = None
    _last_eta: Optional[float] = None
    _last_n_max: Optional[int] = None
    _rj: float
    _rbarj: float
    _rest_cj: float

    def __init__(self) -> None:
        self._grid = Grid(no_zero=True)

    @staticmethod
    @mx.compile
    def _sum_kernel_rest(
        n_norm: mx.array,
        j: float,
        eta: float,
    ) -> mx.array:
        """MLX kernel for the rest-frame sum"""
        t = mx.tanh(mx.sinh(eta * n_norm)) ** (j + 2.0)
        return mx.sum((1.0 - t) / (n_norm**j))

    @staticmethod
    @mx.compile
    def _residual_kernel(
        n_norm: mx.array,
        n_hat: mx.array,
        v: mx.array,
        j: float,
        eta: float,
        a: float,
    ) -> mx.array:
        """MLX kernel for the rest-frame/moving-frame residual sum"""
        p = j + 2.0
        ndv = n_hat @ v
        d = 1.0 / (1.0 - ndv)
        tv = mx.tanh(mx.sinh(eta * n_norm * d ** (-1.0 / p))) ** p
        t0 = mx.tanh(mx.sinh(eta * n_norm)) ** p
        term_v = (1.0 - tv) * d / (n_norm**j)
        term_0 = (1.0 - t0) / (n_norm**j)
        return mx.sum(term_v - a * term_0)

    def _refresh_cache(
        self,
        j: float,
        eta: float,
        n_max: int,
        n_norm: mx.array,
        *,
        device: mx.DeviceType,
    ) -> None:
        """
        Function caching `R_j` [2, Eq. (A26)], `Rbar_j` [2, Eq. (A33)], and `c_j(0)`
        (using [2, Eqs. (A25) & (A32)]).
        """
        refresh_j = self._last_j is None or j != self._last_j
        refresh_rest = (
            refresh_j
            or self._last_eta is None
            or eta != self._last_eta
            or n_max != self._last_n_max
        )

        # refresh R_j and Rbar_j if j changed
        if refresh_j:
            if j > 3.0:
                self._rj = 0.0
                self._rbarj = _rbarj(j)
            elif j < 3.0:
                self._rj = _rj(j)
                if j < 0.0:
                    self._rbarj = _rbarj(3.0 - j)
                else:
                    self._rbarj = 0.0
            else:
                self._rj = 0.0
                self._rbarj = 0.0

        # refresh c_j(0) and j/eta/n_max changed
        if refresh_rest:
            with mx.stream(mx.Device(device)):
                # use more stable reflection formula [1, Eq. (70)] for j < 0
                if j < 0.0:
                    jp = 3.0 - j
                    s0 = self._sum_kernel_rest(n_norm, jp, eta).item()
                    c0p = s0 + 4.0 * math.pi * eta ** (jp - 3.0) * self._rbarj
                    self._rest_cj = self._reflect_cj(j, c0p)
                # use [2, Eq. (A25)] for j < 3
                elif j < 3.0:
                    s0 = self._sum_kernel_rest(n_norm, j, eta).item()
                    self._rest_cj = s0 - 4.0 * math.pi * eta ** (j - 3.0) * self._rj
                # use [2, Eq. (A33)] for j > 3
                elif j > 3.0:
                    s0 = self._sum_kernel_rest(n_norm, j, eta).item()
                    self._rest_cj = s0 + 4.0 * math.pi * eta ** (j - 3.0) * self._rbarj
                else:
                    raise RuntimeError("not implemented")

        if refresh_j:
            self._last_j = j
        if refresh_rest:
            self._last_eta = eta
            self._last_n_max = n_max

    @staticmethod
    def _reflect_cj(j: float, c_3_minus_j: float) -> float:
        """reflection formula [1, Eq. (70)]"""
        return (
            math.pi ** (j - 1.5)
            * math.gamma((3.0 - j) / 2.0)
            / math.gamma(j / 2.0)
            * c_3_minus_j
        )

    @staticmethod
    def _v_fp64(v: ArrayLike) -> float:
        return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))

    def __call__(
        self,
        j: float,
        v: ArrayLike = np.zeros(3),
        n_max: Optional[int] = None,
        eta: Optional[float] = None,
        *,
        device: mx.DeviceType,
    ) -> float:
        """Compute c_j(v)"""
        if n_max is None or eta is None:
            raise RuntimeError("not implemented")

        with mx.stream(mx.Device(device)):
            # evaluate |v| in FP64 to avoid rounding issues in functions of |v|
            beta = self._v_fp64(v)
            v_mlx = mx.array(np.asarray(v, dtype=np.float32))

            # set the momentum lattice
            self._grid.n_max = n_max
            n_norm = self._grid.n_norm
            n_hat = self._grid.n_hat

            # compute & cache R_j, Rbar_j, and c_j(0) as needed
            self._refresh_cache(j, eta, n_max, n_norm, device=device)

            # compute A_{5/(j+2)}(|v|)
            k = 5.0 / (j + 2.0)
            a = ak(k, beta)

            # compute c_j(v) - c_j(0) if required
            if beta == 0.0:
                s = 0.0
            else:
                s = self._residual_kernel(n_norm, n_hat, v_mlx, j, eta, a).item()
            if j != 3.0:
                return s + a * self._rest_cj
            else:
                raise RuntimeError("not implemented")
