import unittest
from dataclasses import dataclass

import mlx.core as mx

from hyperzeta import AccelerationParameters, QedCoef


@dataclass(frozen=True)
class CoefCase:
    j: float
    v: tuple[float, ...]
    residual: float
    expected: float


@dataclass(frozen=True)
class FixedCoefCase:
    j: float
    v: tuple[float, ...]
    par: AccelerationParameters
    expected: float


# values checked against papers and https://github.com/aportelli/QedFvCoef
CASES = [
    CoefCase(j=-3.0, v=(0.0, 0.0, 0.0), residual=1.0e-8, expected=0.0411832525),
    CoefCase(j=-1.0, v=(0.0, 0.0, 0.0), residual=1.0e-8, expected=-0.26659628),
    CoefCase(j=0.0, v=(0.0, 0.0, 0.0), residual=1.0e-8, expected=-1.0),
    CoefCase(j=1.0, v=(0.0, 0.0, 0.0), residual=1.0e-8, expected=-2.8372975),
    CoefCase(j=2.0, v=(0.0, 0.0, 0.0), residual=1.0e-8, expected=-8.913633),
    CoefCase(j=4.0, v=(0.0, 0.0, 0.0), residual=1.0e-8, expected=16.532316),
    CoefCase(j=5.0, v=(0.0, 0.0, 0.0), residual=1.0e-8, expected=10.3775248),
    CoefCase(j=2.0, v=(0.9, 0.0, 0.0), residual=1.0e-8, expected=-10.795283),
    CoefCase(j=0.0, v=(0.4, 0.9, 0.1), residual=1.0e-8, expected=-16.159134),
    CoefCase(j=3.0, v=(0.0, 0.0, 0.0), residual=1.0e-8, expected=3.8219235),
    CoefCase(j=3.0, v=(0.2713834, 0.0, 0.0), residual=1.0e-8, expected=3.9238834),
    CoefCase(j=3.0, v=(0.15668327,) * 3, residual=1.0e-8, expected=3.9176440),
]

CPU_FIXED_CASES = [
    FixedCoefCase(
        j=-1.0,
        v=(0.0, 0.0, 0.0),
        par=AccelerationParameters(n_max=10, eta=0.56),
        expected=-0.2665974229838872,
    ),
    FixedCoefCase(
        j=2.0,
        v=(0.0, 0.0, 0.0),
        par=AccelerationParameters(n_max=15, eta=0.53),
        expected=-8.913595314474906,
    ),
    FixedCoefCase(
        j=2.0,
        v=(0.9, 0.0, 0.0),
        par=AccelerationParameters(n_max=35, eta=0.53),
        expected=-10.795528282040351,
    ),
]

GPU_FIXED_CASES = [
    FixedCoefCase(
        j=-1.0,
        v=(0.0, 0.0, 0.0),
        par=AccelerationParameters(n_max=10, eta=0.56),
        expected=-0.2665974286620929,
    ),
    FixedCoefCase(
        j=2.0,
        v=(0.0, 0.0, 0.0),
        par=AccelerationParameters(n_max=25, eta=0.53),
        expected=-8.91359425008389,
    ),
    FixedCoefCase(
        j=2.0,
        v=(0.9, 0.0, 0.0),
        par=AccelerationParameters(n_max=35, eta=0.53),
        expected=-10.795532921470183,
    ),
]


class TestQedCoef(unittest.TestCase):
    def test_qed_coef_cpu_fixed(self) -> None:
        coef = QedCoef()
        for case in CPU_FIXED_CASES:
            with self.subTest(j=case.j, v=case.v):
                cj_single = coef(
                    case.j,
                    case.v,
                    case.par,
                    device=mx.cpu,
                    dtype=mx.float64,
                    n_threads=1,
                )
                cj_multi = coef(
                    case.j,
                    case.v,
                    case.par,
                    device=mx.cpu,
                    dtype=mx.float64,
                    n_threads=2,
                )
                self.assertAlmostEqual(cj_single, case.expected, delta=1.0e-8)
                self.assertAlmostEqual(cj_multi, case.expected, delta=1.0e-8)
                self.assertAlmostEqual(cj_single, cj_multi, delta=1.0e-10)

    def test_qed_coef_gpu_fixed(self) -> None:
        if not mx.is_available(mx.Device(mx.gpu)):
            self.skipTest("GPU is not available")

        coef = QedCoef()
        for case in GPU_FIXED_CASES:
            with self.subTest(j=case.j, v=case.v):
                cj = coef(
                    case.j,
                    case.v,
                    case.par,
                    device=mx.gpu,
                    dtype=mx.float32,
                    n_threads=1,
                )
                self.assertAlmostEqual(cj, case.expected, delta=1.0e-4)

    def test_qed_coef(self) -> None:
        coef = QedCoef()
        for case in CASES:
            with self.subTest(j=case.j, v=case.v, residual=case.residual):
                par = coef.tune(
                    case.j,
                    case.v,
                    case.residual,
                    device=mx.cpu,
                    dtype=mx.float64,
                    n_threads=8,
                )
                cj = coef(
                    case.j,
                    case.v,
                    par,
                    device=mx.cpu,
                    dtype=mx.float64,
                    n_threads=8,
                )
                delta = abs(case.residual * case.expected)
                self.assertAlmostEqual(cj, case.expected, delta=delta)


if __name__ == "__main__":
    unittest.main()
