#ifndef MPI_IMPL_LAPLACIAN_HPP
#define MPI_IMPL_LAPLACIAN_HPP

#include <concepts>

#include "types.hpp"
#include "glaplace.hpp"
#include "mpi_impl/Mesh.hpp"
#include "mpi_impl/comm_base.hpp"
#include "mpi_impl/mask.hpp"
#include "mpi_impl/unmask.hpp"
#include "mpi_impl/global_sum.hpp"

namespace schro_mpi
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
    SparseData<matrix<real>> galerkin_op(Op op, SparseData<matrix<real>> u, const Mesh<real>& mesh)
    {
        unmask<real>(u, mesh);
        
        op(u, mesh);
        
        global_sum<real>(u, mesh);
        mask<real>(u, mesh);
        
        return u;
    }

    // computes the inner product (u, v).
    // maybe use higher order quadrature rule?
    // algorithm 133 in:
    // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
    // Equations: Algorithms for Scientists and Engineers. Scientific computation.
    // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
    template <std::floating_point real>
    SparseData<matrix<real>> gproj(SparseData<matrix<real>> u, const Mesh<real>& mesh)
    {
        unmask<real>(u, mesh);
        
        const auto& w = mesh.quadrature.w;
        for (auto& [el, values] : u)
            values = ( arma::diagmat(w) * values * arma::diagmat(w) ) % mesh.elements.at(el).J;

        global_sum<real>(u, mesh);
        mask<real>(u, mesh);

        return u;
    }
} // namespace schro_mpi


#endif