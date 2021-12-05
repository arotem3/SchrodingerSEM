#ifndef OMP_IMPL_POISSON_HPP
#define OMP_IMPL_POISSON_HPP

#include <concepts>

#include "types.hpp"
#include "pcg.hpp"
#include "openmp_impl/galerkin.hpp"
#include "openmp_impl/solution_wrapper.hpp"
#include "openmp_impl/dot.hpp"

namespace schro_omp
{
    // solves the poisson equation: Lu == f where L is the NEGATIVE laplacian
    // operator with Dirichlet boundary conditions (assumed to be already
    // enforced on u).
    template <std::floating_point real>
    solver_results<real> poisson (std::vector<matrix<real>>& u, const std::vector<matrix<real>>& f, const Mesh<real>& mesh, int max_iter, real tol)
    {
        solution_wrapper<real> b(gproj(f, mesh)); // inner product (f, v)

        solution_wrapper<real> x(std::move(u));

        // computes the inner produce (grad u, grad v) associated with the inner
        // produce (lap u, v) {they are equivalent with dirichlet boundary
        // conditions}
        // algorithm 133 in:
        // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
        // Equations: Algorithms for Scientists and Engineers. Scientific computation.
        // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
        auto L = [&](const solution_wrapper<real>& v) -> solution_wrapper<real>
        {
            return solution_wrapper<real>
            (
                galerkin_op<real>
                (
                    [](std::vector<matrix<real>>& u, const Mesh<real>& mesh) -> void
                    {
                        #pragma omp parallel for if(u.size() > 63) schedule(dynamic) 
                        for (int i=0; i < u.size(); ++i)
                            u[i] = glaplace(u[i], mesh.elements[i], mesh.D, mesh.quadrature);
                    },
                    v.values,
                    mesh
                )
            );
        };

        auto dotprod = [&](const solution_wrapper<real>& u1, const solution_wrapper<real>& u2) -> real
        {
            return dot<real>(mesh, u1.values, u2.values);
        };

        solver_results<real> rslts = pcg<real>(x, L, b, dotprod, IdentityPreconditioner{}, max_iter, tol);

        u = std::move(x.values);

        return rslts;
    }
} // namespace schro_omp


#endif