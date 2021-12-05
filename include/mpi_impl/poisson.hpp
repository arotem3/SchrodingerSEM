#ifndef MPI_IMPL_POISSON_HPP
#define MPI_IMPL_POISSON_HPP

#include <concepts>

#include "types.hpp"
#include "pcg.hpp"
#include "mpi_impl/galerkin.hpp"
#include "mpi_impl/solution_wrapper.hpp"
#include "mpi_impl/dot.hpp"

namespace schro_mpi
{
    // solves -(u_xx + u_yy) == f(x,y) with dirichlet Boundary conditions,
    // (assume to already be enforced on u)
    template <std::floating_point real>
    solver_results<real> poisson(SparseData<matrix<real>>& u, const SparseData<matrix<real>>& f, const Mesh<real>& mesh, mpi::communicator& comm, const std::unordered_map<int,int>& E2P, int max_iter, real tol)
    {
        solution_wrapper<real> b(gproj(f, mesh, comm, E2P)); // inner product (f, v)

        solution_wrapper<real> x(std::move(u));

        // computes the inner produce (grad u, grad v) associated with the inner
        // produce (lap u, v) {they are equivalent with dirichlet boundary
        // conditions}
        // algorithm 133 in:
        // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
        // Equations: Algorithms for Scientists and Engineers. Scientific computation.
        // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
        auto L = [&comm, &mesh, &E2P](const solution_wrapper<real>& v) -> solution_wrapper<real>
        {
            return solution_wrapper<real>
            (
                galerkin_op<real>
                (
                    [](SparseData<matrix<real>>& u, const Mesh<real>& mesh, mpi::communicator& comm, const std::unordered_map<int,int>& E2P) -> void
                    {
                        for (auto& [el, values] : u)
                            values = glaplace<real>(values, mesh.elements.at(el), mesh.D, mesh.quadrature);
                    },
                    v.values,
                    mesh,
                    comm,
                    E2P
                )
            );
        };

        auto dotprod = [&comm, &mesh](const solution_wrapper<real>& x, const solution_wrapper<real>& y) -> real
        {
            return dot(comm, mesh, x.values, y.values);
        };

        solver_results<real> rslts = pcg<real>(x, L, b, dotprod, IdentityPreconditioner{}, max_iter, tol);

        u = std::move(x.values);

        return rslts;
    }

} // namespace schro_mpi


#endif