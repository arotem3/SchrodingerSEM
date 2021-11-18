#ifndef OMP_IMPL_GALERKIN_HPP
#define OMP_IMPL_GALERKIN_HPP

#include <concepts>
#include <vector>

#include "types.hpp"
#include "glaplace.hpp"
#include "openmp_impl/Mesh.hpp"
#include "openmp_impl/mask.hpp"
#include "openmp_impl/unmask.hpp"
#include "openmp_impl/global_sum.hpp"

namespace schro_omp
{
    // computes the inner produce (grad u, grad v) associated with the inner
    // produce (lap u, v) {they are equivalent with dirichlet boundary
    // conditions}
    // algorithm 133 in:
    // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
    // Equations: Algorithms for Scientists and Engineers. Scientific computation.
    // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
    template <std::floating_point real>
    std::vector<matrix<real>> laplacian(std::vector<matrix<real>> u, const Mesh<real>& mesh)
    {
        unmask<real>(u, mesh);

        #pragma omp parallel for if(u.size() > 63) schedule(dynamic) 
        for (int i=0; i < u.size(); ++i)
            u[i] = glaplace<real>(u[i], mesh.elements[i], mesh.D, mesh.quadrature);
        
        
        global_sum<real>(u, mesh);
        mask<real>(u, mesh);

        return u;
    }

    // computes the inner produce (c*u, v) where c may vary spatially
    // algorithm 133 in:
    // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
    // Equations: Algorithms for Scientists and Engineers. Scientific computation.
    // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
    template <std::floating_point real>
    std::vector<matrix<real>> scale(const std::vector<matrix<real>>& c, std::vector<matrix<real>> u, const Mesh<real>& mesh)
    {
        unmask<real>(u, mesh);

        const auto& w = mesh.quadrature.w;

        #pragma omp parallel for if(u.size() > 63) schedule(dynamic) 
        for (int e=0; e < u.size(); ++e)
            u[e] = ( arma::diagmat(w) * (u[e] % c[e]) * arma::diagmat(w) ) % mesh.elements[e].J;
        
        global_sum<real>(u, mesh);
        mask<real>(u, mesh);

        return u;
    }

    // computes the inner produce (u, v)
    // algorithm 133 in:
    // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
    // Equations: Algorithms for Scientists and Engineers. Scientific computation.
    // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
    template <std::floating_point real>
    std::vector<matrix<real>> identity(std::vector<matrix<real>> u, const Mesh<real>& mesh)
    {
        unmask<real>(u, mesh);

        const auto& w = mesh.quadrature.w;

        #pragma omp parallel for if(u.size() > 63) schedule(dynamic) 
        for (int e=0; e < u.size(); ++e)
            u[e] = (arma::diagmat(w) * u[e] * arma::diagmat(w)) % mesh.elements[e].J;

        global_sum<real>(u, mesh);
        mask<real>(u, mesh);

        return u;
    }

} // namespace schro_omp


#endif