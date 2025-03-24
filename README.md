# ReplicaSim

**ReplicaSim** is a GPU-accelerated simulation tool written in Julia and using low-level CUDA kernel programming for performing large-scale Monte Carlo simulations of **artificial spin ice (ASI)** systems. It is designed to efficiently simulate multiple replicas in parallel, making it suitable for high-throughput data generation in computational nanomagnetism research.

This tool was developed as part of my MSc research project:  
**"Localized Control of Magnetization Reversal in Square Artificial Spin Ice"**  
_Department of Physics and Astronomy, University of Manitoba, 2024_

---

## Features

- Custom CUDA kernel programming using [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
- Efficient simulation of thousands of replicas simultaneously
- Designed for HPC environments with batch-job compatibility
- Open and transparent for reproducibility and peer review

📄 [Read the full technical description and motivation (PDF)](ReplicaSim_Description.pdf)

---

## Getting Started

### Requirements
- Julia ≥ 1.11
- CUDA-enabled GPU
- Julia packages:
  - `CUDA.jl`
  - `LinearAlgebra`
  - `JLD2`
  - (Optional) `BenchmarkTools.jl`, `Plots.jl`, etc.

### Install Dependencies
```julia
using Pkg
Pkg.add("CUDA")
```


### License
This project is licensed under the MIT License.



### For Reviewers
This repository is intended to provide full transparency for the simulation code described in the submitted manuscript. For any questions or reproducibility concerns, feel free to open an issue or contact me directly.



### Acknowledgements
- HPC access provided by **Digital Alliance Canada**
- Research supported by **University of Manitoba**, **NSERC**, and **Canadian Foundation for Innovation (CFI) John R. Evans Leaders Fund**
- Supervised by Prof. Robert L. Stamps and Prof. Johan van Lierop




