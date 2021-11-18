#ifndef OMP_IMPL_MASK_HPP
#define OMP_IMPL_MASK_HPP

#include "types.hpp"
#include "mask_side.hpp"
#include "mask_node.hpp"
#include "openmp_impl/Mesh.hpp"

namespace schro_omp
{
    // set all redundant degrees of freedom to zero.
    // algorithm 130 in:
    // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
    // Equations: Algorithms for Scientists and Engineers. Scientific computation.
    // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
    template <std::floating_point real>
    void mask(std::vector<matrix<real>>& a, const Mesh<real>& mesh)
    {
        auto always_true = [](int e) -> bool {return true;};

        #pragma omp parallel for if(mesh.edges.size() > 63) schedule(dynamic) 
        for (auto it = mesh.edges.cbegin(); it != mesh.edges.cend(); ++it)
            mask_side(a, *it, mesh, always_true);

        #pragma omp parallel for if(mesh.edges.size() > 63) schedule(dynamic) 
        for (auto it = mesh.nodes.cbegin(); it != mesh.nodes.cend(); ++it)
            mask_node(a, *it, always_true);
    }
} // namespace schro_omp

#endif