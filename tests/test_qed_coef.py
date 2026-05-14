import unittest
from dataclasses import dataclass

import mlx.core as mx

from hyperzeta import QedCoef


@dataclass(frozen=True)
class CoefCase:
    j: float
    v: tuple[float, float, float]
    residual: float
    expected: float


CASES = [
    CoefCase(j=-3.0, v=(0.0, 0.0, 0.0), residual=1.0e-4, expected=0.04118),
    CoefCase(j=-1.0, v=(0.0, 0.0, 0.0), residual=1.0e-4, expected=-0.26660),
    CoefCase(j=0.0, v=(0.0, 0.0, 0.0), residual=1.0e-4, expected=-1.0),
    CoefCase(j=1.0, v=(0.0, 0.0, 0.0), residual=1.0e-4, expected=-2.83730),
    CoefCase(j=2.0, v=(0.0, 0.0, 0.0), residual=1.0e-4, expected=-8.91363),
    CoefCase(j=4.0, v=(0.0, 0.0, 0.0), residual=1.0e-4, expected=16.53232),
    CoefCase(j=5.0, v=(0.0, 0.0, 0.0), residual=1.0e-4, expected=10.37752),
    CoefCase(j=2.0, v=(0.9, 0.0, 0.0), residual=1.0e-4, expected=-1.07952828825e01),
    CoefCase(j=0.0, v=(0.4, 0.9, 0.1), residual=1.0e-3, expected=-1.61591340961e01),
]


class TestQedCoef(unittest.TestCase):
    def test_qed_coef(self) -> None:
        coef = QedCoef()
        coef.log = True
        for case in CASES:
            with self.subTest(j=case.j, v=case.v, residual=case.residual):
                par = coef.tune(case.j, case.v, case.residual, device=mx.gpu)
                cj = coef(case.j, case.v, par, device=mx.gpu)
                delta = abs(case.residual * case.expected)
                self.assertAlmostEqual(cj, case.expected, delta=delta)


if __name__ == "__main__":
    unittest.main()
