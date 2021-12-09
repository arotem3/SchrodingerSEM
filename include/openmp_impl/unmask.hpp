#ifndef OMP_IMPL_UNMASK_HPP
#define OMP_IMPL_UNMASK_HPP

#include "openmp_impl/Mesh.hpp"
#include "get_edge_values.hpp"
#include "set_edge_values.hpp"

namespace schro_omp
{
    // match the values of two elements on a side according to the element
    // managing that side.
    template <std::floating_point real>
    void unmask_side(std::vector<matrix<real>>& a, const Edge& edge, const Mesh<real>& mesh)
    {    
        int e0 = edge.elements[0]; // the element managing this edge
        int s0 = edge.element_sides[0];

        std::vector<real> edge_values = get_edge_values<real>(a, mesh, e0, s0);

        int e1 = edge.elements[1];
        int s1 = std::abs(edge.element_sides[1]);

        set_edge_values<real>(a, mesh, edge, edge_values, e1, s1);
    }

    // match the values of all elements sharing a node according to the element
    // managing that node.
    template <std::floating_point real>
    void unmask_node(std::vector<matrix<real>>& a, const CornerNode<real>& node, const Mesh<real>& mesh)
    {
        auto el = node.connected_elements.cbegin();
        const real cval = a[el->element_id].at(el->i, el->j);
        ++el;
        for (; el != node.connected_elements.cend(); ++el)
            a[el->element_id].at(el->i, el->j) = cval;
    }

    // match values at element interfaces.
    // algorithm 131 in:
    // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
    // Equations: Algorithms for Scientists and Engineers. Scientific computation.
    // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
    template <std::floating_point real>
    void unmask(std::vector<matrix<real>>& a, const Mesh<real>& mesh)
    {
        #pragma omp parallel for if(mesh.edges.size() > 63) schedule(static) 
        for (auto it = mesh.edges.cbegin(); it != mesh.edges.cend(); ++it)
            if (it->edge_type != BOUNDARY)
                unmask_side(a, *it, mesh);
        
        #pragma omp parallel for if(mesh.nodes.size() > 63) schedule(static) 
        for (auto it = mesh.nodes.cbegin(); it != mesh.nodes.cend(); ++it)
            if (it->corner_type != BOUNDARY)
                unmask_node(a, *it, mesh);
    }
    
} // namespace schro_omp


#endif