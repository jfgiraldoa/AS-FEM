# ASFEM: Adaptive Stabilized Finite Element Method 

ASFEM is a computational framework for solving partial differential equations using **adaptive stabilized finite element methods**. It is designed to address challenging linear and nonlinear **advection–diffusion–reaction** problems, particularly in **advection-dominated regimes**, where standard methods often suffer from instability and poor accuracy.

The method integrates a **residual minimization strategy** that couples a **semi-discrete discontinuous Galerkin (DG) formulation in space** with a **time-marching scheme** based on the method of lines. This combination yields a **stable, continuous solution** while simultaneously providing an **on-the-fly error estimator** that effectively guides **adaptive mesh refinement at each discrete time step**.

ASFEM has been tested across a range of problems, including benchmark cases for validation, highly convective transport scenarios to demonstrate stability, and nonlinear systems that showcase the method’s robustness and flexibility.

The current framework includes several benchmark problems:
- Eriksson–Johnson problem (steady and transient)
- Nonlinear Burgers' equation
- 3D Fichera corner problem
- Highly heterogeneous media
- Highly anisotropic diffusion cases

## Features

- Adaptive refinement driven by residual-based error estimators
- Stabilized finite element formulations for convection-dominated problems
- Coupling of DG spatial discretization with time-marching schemes
- Modular structure for easy experimentation and extension
- Benchmarks and test cases for validation and demonstration

For more information about the methodology and applications, please refer to the reference papers below.

## Installation (macOS Apple Silicon, Legacy FEniCS 2019)

> **Note:** This setup must be run under Rosetta (x86_64).

### Step 1: Open Terminal with Rosetta

- Go to `Applications > Utilities`.
- Right-click Terminal → **Get Info**.
- Check **Open using Rosetta**.
- Close and reopen Terminal.

### Option 1: Using Miniforge3 (Recommended)

```bash
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
bash Miniforge3-MacOSX-x86_64.sh
```

After installation, restart the Terminal (still using Rosetta):

```bash
conda --version  # Confirm installation
```

### Option 2: Manual Environment Setup

1. **Install TBB via Intel Homebrew**
   ```bash
   brew install tbb
   ```

2. **Create and activate the Conda environment (Intel mode)**
   ```bash
   CONDA_SUBDIR=osx-64 conda create -n asfem64 python=3.9 -y
   conda activate asfem64
   conda config --env --set subdir osx-64
   ```

3. **Install FEniCS 2019 and required packages**
   ```bash
   conda install -c defaults -c conda-forge \
       fenics matplotlib numpy sympy scipy scikit-sparse meshio h5py mpi4py -y
   pip install spb pygmsh mpltools
   ```

Your legacy FEniCS 2019 environment is now ready on macOS Apple Silicon.

## Folder Structure

```bash
ASFEM/
├── Benchmark/           # Test cases and reference problems
├── README.md            # Project overview, documentation, and installation
├── src/                 # Source code for ASFEM
├── tests/               # Unit tests
```

## Getting Started

To run a benchmark or unit test:

```bash
cd ASFEM
python tests/unittest_run.py  
```

## License

This project is distributed under the MIT License, a permissive open-source license that allows reuse, modification, and distribution with proper attribution.  
© 2025 Juan Felipe Giraldo. See the LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to open issues, submit pull requests, or suggest new features.

## References

1. Giraldo, J. F., & Calo, V. M. (2023). An Adaptive in Space, Stabilized Finite Element Method via Residual Minimization for Linear and Nonlinear Unsteady Advection–Diffusion–Reaction Equations. *Mathematical and Computational Applications*, 28(1), 7. [https://doi.org/10.3390/mca28010007](https://doi.org/10.3390/mca28010007)

2. Giraldo, J. F., & Calo, V. M. (2023). A variational multiscale method derived from an adaptive stabilized conforming finite element method via residual minimization on dual norms. *Computer Methods in Applied Mechanics and Engineering*, 417, 116285. [https://doi.org/10.1016/j.cma.2023.116285](https://doi.org/10.1016/j.cma.2023.116285)
