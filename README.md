# Traveling-wave solutions and structure-preserving numerical methods for a hyperbolic approximation of the Korteweg-de Vries equation

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14423351.svg)](https://doi.org/10.5281/zenodo.14423351)


This repository contains information and code to reproduce the results
presented in the article
```bibtex
@article{biswas2025traveling,
  title={Traveling-wave solutions and structure-preserving numerical methods
         for a hyperbolic approximation of the {K}orteweg-de {V}ries equation},
  author={Biswas, Abhijit and Ketcheson, David I. and Ranocha, Hendrik and
          Sch{\"u}tz, Jochen},
  journal={Journal of Scientific Computing},
  volume={103},
  pages={90},
  year={2025},
  month={05},
  doi={10.1007/s10915-025-02898-x},
  eprint={2412.17117},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{biswas2024travelingRepro,
  title={Reproducibility repository for
         "{T}raveling-wave solutions and structure-preserving numerical methods
         for a hyperbolic approximation of the {K}orteweg-de {V}ries equation"},
  author={Biswas, Abhijit and Ketcheson, David I. and Ranocha, Hendrik and
          Sch{\"u}tz, Jochen},
  year={2024},
  howpublished={\url{https://github.com/abhibsws/2024_kdvh_RR}},
  doi={10.5281/zenodo.14423351}
}
```


## Abstract

We study the recently-proposed hyperbolic approximation of the Korteweg-de Vries equation (KdV).
We show that this approximation, which we call KdVH, possesses a rich variety of
solutions, including solitary wave solutions that approximate KdV solitons, as well as other
solitary and periodic solutions that are related to higher-order water wave models,
and may include singularities.
We analyze a class of implicit-explicit Runge-Kutta time discretizations for KdVH
that are asymptotic preserving, energy conserving, and can be applied to other hyperbolized
systems. We also develop structure-preserving spatial discretizations based on summation-by-parts
operators in space including finite difference, discontinuous Galerkin, and Fourier methods. We use the
relaxation approach to make the fully discrete schemes energy-preserving.
Numerical experiments demonstrate the effectiveness of these discretizations.


## Numerical experiments

The numerical experiments use [Python](https://www.python.org)
and [Julia](https://julialang.org).

To run the Python code, you need Python 3, and the packages
`jupyter`, `numpy`, `scipy`, and `matplotlib`.
The Python code has been tested with the following versions, but
other versions may work:

    - Python 3.13
    - Jupyter 1.1.1
    - Numpy 2.1.3
    - SciPy 1.14.1
    - Matplotlib 3.9.2

We have used Julia version 1.10.7 for the experiments. The results can be reproduced
by running all cells of the Jupyter notebook `figures_manuscript.ipynb` in order.


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
