# Finite Difference - Incompressible Navier Stokes

<p align="right"> CMSE 822 <br> Amit Rotem </p>

## Summary
<html>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>
<p>
This project will implement a finite difference simulation of the incompressible Navier-Stokes equations on a domain \(\Omega\subset\mathbb{R}^2\):
\[
\begin{align*}
\mathbf{u}_t + (\mathbf{u}\cdot\nabla)\mathbf{u} + \nabla p &= \nu \Delta \mathbf{u} + \mathbf{F}, \qquad & \mathbf{x}&\in\Omega,\; t > 0 \\
\nabla \cdot \mathbf{u} &= 0,\qquad & \mathbf{x}&\in\Omega, \; t > 0 \\
\mathcal{B}(\mathbf{u}, p) &= \mathbf{g}, \qquad & \mathbf{x}&\in\partial\Omega,\; t > 0 \\
\mathbf{u}(\mathbf{x},0) &= \mathbf{u}_0(\mathbf{x}), \qquad & \mathbf{x}&\in\Omega.
\end{align*}
\]
The time discretization will be implemented using an implicit-explicit (IMEX) backwards differentiation scheme, so each time step requires solving a large sparse linear system of equations. To solve this system, we can use the conjugate gradient method. The benefit of this method which solely requires matrix-vector products (rather than explicitly constructing and factoring the matrix). Luckily, the finite difference operators use very localized information leading to natural parallelization in both shared and distributed memory paradigms.
</p>
</html>

## Parallelization Strategies
The main goal of this project is to translate an <a href="https://github.com/appelo/CMSE_823_2021/blob/main/project/code/ins.f90">existing finite difference code</a> to C++ and add both MPI and OpenMP parallelism. I am mainly interested in using the <a href="https://www.boost.org/doc/libs/1_77_0/doc/html/mpi.html">Boost MPI library</a> for distributing work accross nodes on an HPC cluster and OpenMP for distributing work among cores within each node.

Because finite difference methods are typically restricted to a narrow set of problem geometries and mesh design, load balancing is not likely to pose a challenge. However, the highly structured mesh will enable implementation and exploration of MPI communicator topologies.

The finite difference operations are so local, in fact, that GPU offloading may be an even better avenue for parallelism than OpenMP multithreading.

## Validation and Optimization
I intend to compare the implementation to the existing serial implementation and analyze speedup and efficiency. It will be interesting to determine an optimal number of OpenMP threads and MPI nodes for a given problem size. 