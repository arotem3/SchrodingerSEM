#ifndef MPI_IMPL_MASK_HPP
#define MPI_IMPL_MASK_HPP

#include "types.hpp"
#include "mask_side.hpp"
#include "mask_node.hpp"
#include "mpi_impl/Mesh.hpp"

namespace schro_mpi
{
    // set all redundant degrees of freedom to zero.
    // algorithm 130 in:
    // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
    // Equations: Algorithms for Scientists and Engineers. Scientific computation.
    // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
    template <std::floating_point real>
    void mask(SparseData<matrix<real>>& a, const Mesh<real>& mesh)
    {
        auto referencable_element = [&a](int e) -> bool
        {
            return a.contains(e);
        };

        for (auto& [edge_id, edge] : mesh.edges)
            mask_side(a, edge, mesh, referencable_element);

        for (auto& [node_id, node] : mesh.nodes)
            mask_node(a, node, referencable_element);
    }
} // namespace schro_mpi


#endif