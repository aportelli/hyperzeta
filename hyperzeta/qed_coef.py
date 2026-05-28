import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import mlx.core as mx
import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad, tplquad

from hyperzeta.lattice import Lattice

QED_DEFAULT_ERROR: float = 1.0e-4
QED_DEFAULT_ETAINVSTEP: float = 0.1
QED_DEFAULT_NMAXSTEP: int = 5
QED_DEFAULT_MAX_NMAX: int = 200
QED_DEFAULT_J3_EPS: float = 1.0e-3
_QED_Q3_REST: float = -0.7302886099374888
_QED_REF_TRESHOLD = -0.1


def ak(k: float, v: float) -> float:
    """Return the angular factor `A_k(|v|)` [2, Eqs. (A16) & (A17)]."""
    if not (0.0 <= v < 1.0):
        raise ValueError(f"|v| must satisfy 0 <= |v| < 1 (got {v})")
    if v == 0.0:
        return 1.0
    eps = k - 1.0
    if abs(eps) < 1e-8:
        return math.atanh(v) / v
    a = -eps * math.log1p(-v)
    b = -eps * math.log1p(v)
    return (math.expm1(a) - math.expm1(b)) / (2.0 * v * eps)


def _tanhsinh(r: float) -> float:
    """Evaluate tanh(sinh(r)) with a conservative large-r shortcut to avoid overflow."""
    if r > 40.0:
        return 1.0
    else:
        return math.tanh(math.sinh(r))


def _rj(j: float) -> float:
    """Compute the integral `R_j` [2, Eq. (A26)]."""

    def f(r: float) -> float:
        return (1.0 - _tanhsinh(r) ** (j + 2.0)) / (r ** (j - 2.0))

    result, _ = quad(f, 0.0, math.inf)
    return result


def _rbarj(j: float) -> float:
    """Compute the integral `Rbar_j` [2, Eq. (A33)]."""

    def f(r: float) -> float:
        return _tanhsinh(r) ** (j + 2.0) / (r ** (j - 2.0))

    result, _ = quad(f, 0.0, math.inf)
    return result


def _q3(v: ArrayLike = np.zeros(3), cut: float = 40) -> float:
    """Compute the integral `Q_3(v)` [2, Eq. (A31)]"""

    def f(theta: float, phi: float, r: float) -> float:
        n_hat = np.array([
            np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(theta),
        ])
        d = 1 / (1 - np.dot(n_hat, v))
        t = np.tanh(np.sinh(r * d ** (-1 / 5))) ** 5
        return np.sin(theta) * t * d / r

    i = tplquad(f, 0, cut, 0, 2 * np.pi, 0, np.pi)
    return i[0] - 4 * np.pi * ak(1, np.sqrt(np.dot(v, v))) * np.log(cut)


@dataclass
class AccelerationParameters:
    """Acceleration parameters controlling the regulator and momentum lattice cutoff."""

    n_max: int = -1
    eta: float = -1


class QedCoef:
    """
    Class computing the `c_j(v)` QED finite-volume coefficients described in [1,2]

    References
    ----------
    [1]: [Davoudi et al., PRD 99(3), 114510 (2019)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.99.034510)
    [2]: [Di Carlo et al., PRD 105(7), 074509 (2022)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.074509)
    """

    _lattice: Lattice
    _last_j: Optional[float] = None
    _last_eta: Optional[float] = None
    _last_n_max: Optional[int] = None
    _last_dtype: Optional[mx.Dtype] = None
    _rj: float
    _rbarj: float
    _q3_cache: Dict[str, float] = {}
    _rest_cj: float
    _streams: Dict[mx.DeviceType, List[mx.Stream]]

    log: bool = False

    def __init__(self) -> None:
        self._lattice = Lattice(no_zero=True, dtype=mx.float32)
        self._streams = {}

    @staticmethod
    @mx.compile
    def _sum_kernel_rest(
        n_norm: mx.array,
        j: mx.array,
        eta: mx.array,
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
        j: mx.array,
        eta: mx.array,
        a: mx.array,
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
        *,
        dtype: mx.Dtype,
        device: mx.DeviceType,
        n_threads: int = 1,
    ) -> None:
        """Refresh cached integrals and the rest-frame coefficient as needed."""
        refresh_j = self._last_j is None or j != self._last_j
        refresh_rest = (
            refresh_j
            or self._last_eta is None
            or eta != self._last_eta
            or n_max != self._last_n_max
            or self._last_dtype is None
            or dtype != self._last_dtype
        )

        # refresh R_j and Rbar_j if j changed
        if refresh_j:
            if j > 3.0 + QED_DEFAULT_J3_EPS:
                self._rj = 0.0
                self._rbarj = _rbarj(j)
            elif j < 3.0 - QED_DEFAULT_J3_EPS:
                self._rj = _rj(j)
                if j < _QED_REF_TRESHOLD:
                    self._rbarj = _rbarj(3.0 - j)
                else:
                    self._rbarj = 0.0
            else:
                self._rj = 0.0
                self._rbarj = 0.0

        if refresh_rest:
            self._rest_cj = self._compute_rest(
                j, eta, dtype=dtype, device=device, n_threads=n_threads
            )

        if refresh_j:
            self._last_j = j
        if refresh_rest:
            self._last_eta = eta
            self._last_n_max = n_max
            self._last_dtype = dtype

    def _compute_rest(
        self,
        j: float,
        eta: float,
        *,
        dtype: mx.Dtype,
        device: mx.DeviceType,
        n_threads: int,
    ) -> float:
        """Compute the cached rest-frame coefficient `c_j(0)` for the current settings."""
        # use more stable reflection formula [1, Eq. (70)] for j < _QED_REF_TRESHOLD
        if j < _QED_REF_TRESHOLD:
            jp = 3.0 - j
            s0 = self._eval_rest_sum(
                jp,
                eta,
                dtype=dtype,
                device=device,
                n_threads=n_threads,
            )
            c0p = s0 + 4.0 * math.pi * eta ** (jp - 3.0) * self._rbarj
            return self._reflect_cj(j, c0p)
        # use [2, Eq. (A25)] for j < 3
        elif j < 3.0 - QED_DEFAULT_J3_EPS:
            s0 = self._eval_rest_sum(
                j,
                eta,
                dtype=dtype,
                device=device,
                n_threads=n_threads,
            )
            return s0 - 4.0 * math.pi * eta ** (j - 3.0) * self._rj
        # use [2, Eq. (A33)] for j > 3
        elif j > 3.0 + QED_DEFAULT_J3_EPS:
            s0 = self._eval_rest_sum(
                j,
                eta,
                dtype=dtype,
                device=device,
                n_threads=n_threads,
            )
            return s0 + 4.0 * math.pi * eta ** (j - 3.0) * self._rbarj
        else:
            s0 = self._eval_rest_sum(
                j,
                eta,
                dtype=dtype,
                device=device,
                n_threads=n_threads,
            )
            return s0 + 4.0 * math.pi * np.log(eta) + _QED_Q3_REST

    @staticmethod
    def _reflect_cj(j: float, c_3_minus_j: float) -> float:
        """Reflection formula [1, Eq. (70)]"""
        return (
            math.pi ** (j - 1.5)
            * math.gamma((3.0 - j) / 2.0)
            / math.gamma(j / 2.0)
            * c_3_minus_j
        )

    @staticmethod
    def _v_fp64(v: ArrayLike) -> float:
        """Compute |v| in FP64 to avoid instabilities in angular part."""
        return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))

    def _log(self, msg: str):
        if self.log:
            print(f"[{self.__class__.__name__}] {msg}")

    @staticmethod
    def _validate_execution_options(
        device: mx.DeviceType, dtype: mx.Dtype, n_threads: int
    ) -> int:
        """Validate execution options and normalize unsupported thread settings."""
        if dtype not in (mx.float32, mx.float64):
            raise RuntimeError(f"unsupported dtype {dtype}")
        if dtype == mx.float64 and device != mx.cpu:
            raise RuntimeError("float64 is only supported on CPU")
        if device == mx.gpu and n_threads > 1:
            warnings.warn(
                "multiple threads is neither useful nor stable with GPU, reverting to n_threads = 1",
                RuntimeWarning,
            )
            return 1
        return n_threads

    def _eval_chunked(
        self,
        n_items: int,
        kernel: Callable[[int, int], mx.array],
        *,
        device: mx.DeviceType,
        n_threads: int,
    ) -> float:
        """Evaluate a reduction kernel over pool of MLX streams."""
        if n_threads <= 1:
            return kernel(0, n_items).item()

        step = math.ceil(n_items / n_threads)
        chunks = [(i, min(i + step, n_items)) for i in range(0, n_items, step)]
        streams = self._streams.setdefault(device, [])
        dev = mx.Device(device)
        while len(streams) < len(chunks):
            streams.append(mx.new_stream(dev))
        streams = streams[: len(chunks)]
        partials = []

        for stream, (lo, hi) in zip(streams, chunks):
            with mx.stream(stream):
                part = kernel(lo, hi)
                mx.async_eval(part)
                partials.append(part)

        for stream in streams:
            mx.synchronize(stream)

        return sum(part.item() for part in partials)

    def _eval_rest_sum(
        self,
        j: float,
        eta: float,
        *,
        dtype: mx.Dtype,
        device: mx.DeviceType,
        n_threads: int,
    ) -> float:
        """Evaluate the rest-frame lattice sum for the given j and eta."""
        n_norm = self._lattice.n_norm
        j_mx = mx.array(j, dtype=dtype)
        eta_mx = mx.array(eta, dtype=dtype)
        return self._eval_chunked(
            n_norm.shape[0],
            lambda lo, hi: self._sum_kernel_rest(n_norm[lo:hi], j_mx, eta_mx),
            device=device,
            n_threads=n_threads,
        )

    def _eval_residual_sum(
        self,
        v_raw: ArrayLike,
        j: float,
        eta: float,
        a: float,
        *,
        dtype: mx.Dtype,
        device: mx.DeviceType,
        n_threads: int,
    ) -> float:
        """Evaluate the residual sum (cf. notes)."""
        n_norm = self._lattice.n_norm
        n_hat = self._lattice.n_hat
        v_dtype = np.float64 if dtype == mx.float64 else np.float32
        v = mx.array(np.asarray(v_raw, dtype=v_dtype), dtype=dtype)
        j_mx = mx.array(j, dtype=dtype)
        eta_mx = mx.array(eta, dtype=dtype)
        a_mx = mx.array(a, dtype=v.dtype)
        return self._eval_chunked(
            n_norm.shape[0],
            lambda lo, hi: self._residual_kernel(
                n_norm[lo:hi], n_hat[lo:hi], v, j_mx, eta_mx, a_mx
            ),
            device=device,
            n_threads=n_threads,
        )

    def tune(
        self,
        j: float,
        v: ArrayLike = np.zeros(3),
        residual: float = QED_DEFAULT_ERROR,
        *,
        init_par: AccelerationParameters = AccelerationParameters(n_max=5, eta=1.0),
        step: AccelerationParameters = AccelerationParameters(
            n_max=QED_DEFAULT_NMAXSTEP, eta=QED_DEFAULT_ETAINVSTEP
        ),
        max_n_max: int = QED_DEFAULT_MAX_NMAX,
        device: mx.DeviceType,
        dtype: mx.Dtype = mx.float32,
        n_threads: int = 1,
    ) -> AccelerationParameters:
        """
        Tune the acceleration parameters for the computation of `c_j(v)`.

        `step.eta` is an additive step in `1 / eta`, i.e. the update is
        `1 / eta_new = 1 / eta_old + step.eta`.
        """
        n_threads = self._validate_execution_options(device, dtype, n_threads)
        par = AccelerationParameters(n_max=init_par.n_max, eta=init_par.eta)
        if dtype == mx.float32:
            inner_tol = max(1.0e-2 * residual, 1.0e-7)
        else:
            inner_tol = max(1.0e-2 * residual, 1.0e-14)

        def converge(par: AccelerationParameters) -> float:
            previous = self(j, v, par, device=device, dtype=dtype, n_threads=n_threads)
            while True:
                par.n_max += step.n_max
                if par.n_max > max_n_max:
                    raise RuntimeError(f"maximum n_max {max_n_max} exceeded")
                buf = self(j, v, par, device=device, dtype=dtype, n_threads=n_threads)
                res = abs(buf - previous) / (0.5 * (abs(buf) + abs(previous)))
                previous = buf
                if res <= inner_tol:
                    return previous

        previous = converge(par)
        self._log(f"eta= {par.eta:.4f} nmax= {par.n_max} c_j={previous:.15e}")
        while True:
            par.eta = par.eta / (1.0 + step.eta * par.eta)
            par.n_max = par.n_max - 10 if par.n_max > 10 else par.n_max
            buf = converge(par)
            res = abs(buf - previous) / (0.5 * (abs(buf) + abs(previous)))
            res /= math.expm1(step.eta)
            previous = buf
            self._log(
                f"eta= {par.eta:.4f} nmax= {par.n_max} c_j={previous:.15e} residual= {res:.2e}",
            )
            if res <= residual:
                break

        return par

    def __call__(
        self,
        j: float,
        v: ArrayLike = np.zeros(3),
        par: Optional[AccelerationParameters] = None,
        *,
        device: mx.DeviceType,
        dtype: mx.Dtype = mx.float32,
        n_threads: int = 1,
        autotune_residual: float = QED_DEFAULT_ERROR,
    ) -> float:
        """Compute the finite-volume coefficient c_j(v) for the given parameters."""
        n_threads = self._validate_execution_options(device, dtype, n_threads)
        if par is None:
            par = self.tune(
                j, v, autotune_residual, device=device, dtype=dtype, n_threads=n_threads
            )

        with mx.stream(mx.Device(device)):
            beta = self._v_fp64(v)
            if not (0.0 <= beta < 1.0):
                raise ValueError(f"|v| must satisfy 0 <= |v| < 1 (got {beta})")

            # set the momentum lattice
            self._lattice.dtype = dtype
            self._lattice.n_max = par.n_max

            # compute & cache R_j, Rbar_j, and c_j(0) as needed
            self._refresh_cache(
                j, par.eta, par.n_max, dtype=dtype, device=device, n_threads=n_threads
            )

            # compute residual sum if required
            if beta == 0.0:
                s = 0.0
                a = 1.0
                dq3 = 0.0
            else:
                # compute A_{5/(j+2)}(|v|)
                k = 5.0 / (j + 2.0)
                a = ak(k, beta)
                # compute residual sum
                s = self._eval_residual_sum(
                    v, j, par.eta, a, dtype=dtype, device=device, n_threads=n_threads
                )
                # compute residual Q_3 if needed
                if np.abs(j - 3.0) < QED_DEFAULT_J3_EPS:
                    vs = f"{v}"
                    if vs not in self._q3_cache:
                        self._q3_cache[vs] = _q3(v)
                    dq3 = self._q3_cache[vs] - a * _QED_Q3_REST
                else:
                    dq3 = 0.0
            return s + a * self._rest_cj + dq3
