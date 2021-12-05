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
    // applies op to a solution defined defined on a mesh. Intended for
    // designing systems of linear equations as the values of the result are
    // summed on edges and corners. For example op may be the laplacian
    // operator.
    // algorithm 133 in:
    // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
    // Equations: Algorithms for Scientists and Engineers. Scientific computation.
    // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
    template <std::floating_point real, typename Op>
    std::vector<matrix<real>> galerkin_op(Op op, std::vector<matrix<real>> u, const Mesh<real>& mesh)
    {
        unmask<real>(u, mesh);

        op(u, mesh);
        
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
    std::vector<matrix<real>> gproj(std::vector<matrix<real>> u, const Mesh<real>& mesh)
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