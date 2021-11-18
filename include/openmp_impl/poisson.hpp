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
    template <std::floating_point real>
    solver_results<real> poisson (std::vector<matrix<real>>& u, const std::vector<matrix<real>>& f, const Mesh<real>& mesh, int max_iter, real tol)
    {
        solution_wrapper<real> b(identity(f, mesh)); // inner product (f, v)

        solution_wrapper<real> x(std::move(u));

        auto L = [&](const solution_wrapper<real>& v) -> solution_wrapper<real>
        {
            return solution_wrapper<real>(laplacian<real>(v.values, mesh));
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