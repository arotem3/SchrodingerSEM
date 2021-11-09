# Schordinger SEM

The main purpose of this repo is to implement the spectral element method to solve the linear, variable potential, Schrodinger equation on quadrilateral meshes.
The time stepping is achieved using an implicit method (either SDIRK or BDF, still to be determined), which requires solving a variable coefficient Helmholtz equation at each time step. Therefore this repo also provides a Helmholtz and Poisson solver on quadrilateral meshes which are solved using MINRES and CG, respectively. We implement the solvers using two parallel paradigms, MPI and OpenMP. The MPI implementation is best suited for solving the equations on HPC systems, whereas the OpenMP is intended for smaller shared memory systems, and uses OpenMP 4+ GPU offloading.
