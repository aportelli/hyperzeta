# HyperZeta
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Python package implementing different flavours of multidimensional zeta functions, particularly for use in finite-volume quantum field theory.

The main approach used is a doubly exponential acceleration of lattice sums inspired by [Takahashi-Mori quadrature](https://en.wikipedia.org/wiki/Tanh-sinh_quadrature). Lattice sums are evaluated with high-performance [Apple MLX](https://opensource.apple.com/projects/mlx/) kernels which are particularly suitable for Apple Silicon GPUs. However, HyperZeta also implements a multithreaded CPU version which should be platform-agnostic. In theory MLX supports CUDA GPUs, but this package has not been tested in that context.

The following zeta functions have been implemented so far:
- Finite-volume coefficients $c_j(\mathbf{v})$ from [[Davoudi et al., PRD 99(3), 114510 (2019)]](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.99.034510) & [[Di Carlo et al., PRD 105(7), 074509 (2022)]](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.074509).

## Getting started
[Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you have not already done so. Then, at the root of the repository, execute:
```shell
uv sync
```
This will create a Python environment in `.venv`, which should be used to run code from this repository. This environment can be activated in a shell with
```shell
source .venv/bin/activate
```
Most functions have Python docstrings, and the repository has [example notebooks](./notebooks) as well as [implementation notes](./doc).

## Citation policy
If you use HyperZeta in research that leads to a publication, please cite this repository or the corresponding software release.

If contributors to this repository make a substantial intellectual contribution to your research beyond providing the software, please consider co-authorship in line with standard authorship guidelines.
