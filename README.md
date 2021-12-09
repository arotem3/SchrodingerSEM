# Schordinger SEM

The main purpose of this repo is to implement the spectral element method to
solve the linear, variable potential, Schrodinger equation on quadrilateral
meshes. We implement three time stepping strategies: the second order backwards
differentiation formula, an A-stable fourth order three stage diagonally
implicit Runge-Kutta method, and an L-stable fourth order five stage diagonally
implicit Runge-Kutta method which has a continuous extension, and an embedded
error estimate. Each implicit solve uses the MINRES method. We also provide a
Poisson and Helmholtz solvers since the implicit time step is very similar to
solving a Helmholtz equation, so the generalization is simple. All the solvers
have an OpenMP implementation and an MPI implementation.
