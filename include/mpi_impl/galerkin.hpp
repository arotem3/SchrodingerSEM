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
    // computes the inner produce (grad u, grad v) associated with the inner
    // produce (-lap u, v) {they are equivalent with dirichlet boundary
    // conditions}
    // algorithm 133 in:
    // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
    // Equations: Algorithms for Scientists and Engineers. Scientific computation.
    // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.s
    template <std::floating_point real>
    SparseData<matrix<real>> laplacian(SparseData<matrix<real>> u, const Mesh<real>& mesh, mpi::communicator& comm, const std::unordered_map<int,int>& E2P)
    {
        unmask<real>(u, mesh, comm, E2P);

        for (auto& [el, values] : u)
            values = glaplace<real>(values, mesh.elements.at(el), mesh.D, mesh.quadrature);

        global_sum<real>(u, mesh, comm, E2P);
        mask<real>(u, mesh);

        return u;
    }

    // computes the inner product (c*u, v) where c may vary spatially.
    // maybe use higher order quadrature rule?
    // algorithm 133 in:
    // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
    // Equations: Algorithms for Scientists and Engineers. Scientific computation.
    // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
    template <std::floating_point real>
    SparseData<matrix<real>> scale(const SparseData<matrix<real>>& c, SparseData<matrix<real>> u, const Mesh<real>& mesh, mpi::communicator& comm, const std::unordered_map<int,int>& E2P)
    {
        unmask<real>(u, mesh, comm, E2P);

        const auto& w = mesh.quadrature.w;
        for (auto& [el, values] : u)
            values = ( arma::diagmat(w) * (values % c.at(el)) * arma::diagmat(w) ) % mesh.elements.at(el).J;
        
        global_sum<real>(u, mesh, comm, E2P);
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
    SparseData<matrix<real>> identity(SparseData<matrix<real>> u, const Mesh<real>& mesh, mpi::communicator& comm, const std::unordered_map<int,int>& E2P)
    {
        unmask<real>(u, mesh, comm, E2P);

        const auto& w = mesh.quadrature.w;
        for (auto& [el, values] : u)
            values = ( arma::diagmat(w) * values * arma::diagmat(w) ) % mesh.elements.at(el).J;
        
        global_sum<real>(u, mesh, comm, E2P);
        mask<real>(u, mesh);

        return u;
    }
} // namespace schro_mpi


#endif