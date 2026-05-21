# Notes on QED finite-size coefficients
The implementation in HyperZeta follows Sec. A.4 of [[2]](#2), and this note documents deviations from the paper.

## Erratum Eq. (A24)
Eq. (A24) in [[2]](#2) has the wrong sign in the exponent of the argument of the $\sinh$ function. The correct equation is

$$
f(\mathbf{n})=1-\left(\tanh\{\sinh[|\mathbf{n}|d(\hat{\mathbf{n}};\{\mathbf{v}\})^{-\frac{1}{j+2}}]\}\right)^{j+2}.
$$

This can be validated by comparing to [[1]](#1) Eq. (B3).

## UV-improved convergence of boosted sums
In the $j\neq 3$ case, the summand for the boosted coefficient
$c_j(\mathbf{v})$ is

$$
S(\mathbf{n},\mathbf{v})=
\frac{f(\eta\mathbf{n})}{|\mathbf{n}|^j(1-\mathbf{v}\cdot\hat{\mathbf{n}})},
$$

and its integral over $\mathbf{n}$ is given by

$$
\int\mathrm{d}^3\mathbf{n}S(\mathbf{n},\mathbf{v})=\int\mathrm{d}^3\mathbf{n}\frac{1-\tanh[\sinh(\eta|\mathbf{n}|)]^{j+2}}{|\mathbf{n}|^j(1-\mathbf{v}\cdot\hat{\mathbf{n}})^{\frac{5}{j+2}}},
$$

leading to Eq. (B6) in [[1]](#1) if reduced further. An interesting observation here is that the sum of $S(\mathbf{n},\mathbf{v})$ over $\mathbf{n}$ has the same UV (i.e. large-$|\mathbf{n}|$) behaviour as the integral, and therefore

$$
\sum_{\mathbf{n}\neq\mathbf{0}}[S(\mathbf{n},\mathbf{v})-A_{\frac{5}{j+2}}(|\mathbf{v}|)S(\mathbf{n},\mathbf{0})]
$$

is finite for $\eta\to 0$ even if $j<3$, since the same limit with an integral instead of a sum vanishes by the previous identity. Additionally, the limit $\eta\to 0$ is reached exponentially fast, thanks to the properties of $f$ discussed in [[1]](#1). Now, defining the *residual summand*

$$
\bar{S}(\mathbf{n},\mathbf{v})=S(\mathbf{n},\mathbf{v})-A_{\frac{5}{j+2}}(|\mathbf{v}|)S(\mathbf{n},\mathbf{0}),
$$

$c_j(\mathbf{v})$ is given by

$$
c_j(\mathbf{v})=\sum_{\mathbf{n}\neq\mathbf{0}}\bar{S}(\mathbf{n},\mathbf{v})+A_{\frac{5}{j+2}}(|\mathbf{v}|)c_j(\mathbf{0}).
$$

Thanks to the better UV behaviour of the sum, this last identity is much more stable for GPU FP32 evaluations of $c_j(\mathbf{v})$ at high velocities.

In the $j=3$ case, the strategy above can also be applied to rewrite Eq. (A30) in [[2]](#2) as

$$
c_3(\mathbf{v})=\sum_{\mathbf{n}\neq\mathbf{0}}\bar{S}(\mathbf{n},\mathbf{v})+A_{1}(|\mathbf{v}|)c_3(\mathbf{0})+Q_3(\mathbf{v})-A_{1}(|\mathbf{v}|)Q_3(\mathbf{0}).
$$

The implementation in [`qed_coef.py`](../hyperzeta/qed_coef.py) uses the residual strategy above; the residual summand is implemented in the MLX kernel `_residual_kernel`.

## References
<a id="1">[1]</a>
[Davoudi et al., PRD 99(3), 114510 (2019)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.99.034510)  
<a id="2">[2]</a> 
[Di Carlo et al., PRD 105(7), 074509 (2022)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.074509)  
