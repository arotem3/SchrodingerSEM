#ifndef MPI_IMPL_HELMHOLTZ_HPP
#define MPI_IMPL_HELMHOLTZ_HPP

#include <concepts>

#include "types.hpp"
#include "minres.hpp"
#include "mpi_impl/galerkin.hpp"
#include "mpi_impl/solution_wrapper.hpp"
#include "mpi_impl/dot.hpp"

namespace schro_mpi
{
    // computes the inner produce (grad u, grad v) + (c*u, v) associated
    // with the inner produce (lap u + c*u, v) {they are equivalent with
    // dirichlet boundary conditions}
    // algorithm 133 in:
    // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
    // Equations: Algorithms for Scientists and Engineers. Scientific computation.
    // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
    template <std::floating_point real>
    solver_results<real> helmholtz(SparseData<matrix<real>>& u, const SparseData<matrix<real>>& c, const SparseData<matrix<real>>& f, const Mesh<real>& mesh, int max_iter, real tol)
    {
        solution_wrapper<real> b(gproj(f, mesh));

        solution_wrapper<real> x(std::move(u));

        // computes the inner produce (grad u, grad v) + (c*u, v) associated
        // with the inner produce (lap u + c*u, v) {they are equivalent with
        // dirichlet boundary conditions}
        // algorithm 133 in:
        // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
        // Equations: Algorithms for Scientists and Engineers. Scientific computation.
        // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
        auto A = [&](const solution_wrapper<real>& v) -> solution_wrapper<real>
        {
            return solution_wrapper<real>
            (
                galerkin_op<real>
                (
                    [&c](SparseData<matrix<real>>& u, const Mesh<real>& mesh) -> void
                    {
                        const auto& w = mesh.quadrature.w;
                        for (auto& [el, values] : u)
                        {
                            matrix<real> cuv = (arma::diagmat(w) * (values%c.at(el)) * arma::diagmat(w)) % mesh.elements.at(el).J;
                            values = cuv + glaplace(values, mesh.elements.at(el), mesh.D, mesh.quadrature);
                        }
                    },
                    v.values,
                    mesh
                )
            );
        };

        auto dotprod = [&mesh](const solution_wrapper<real>& x, const solution_wrapper<real>& y) -> real
        {
            return dot(mesh, x.values, y.values);
        };
        

        solver_results<real> rslts = minres<real>(x, A, b, dotprod, IdentityPreconditioner{}, max_iter, tol);

        u = std::move(x.values);

        return rslts;
    }
} // namespace schro_mpi


#endif