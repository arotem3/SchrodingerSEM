#ifndef OMP_IMPL_GLOBAL_SUM_HPP
#define OMP_IMPL_GLOBAL_SUM_HPP

#include <concepts>

#include "types.hpp"
#include "get_edge_values.hpp"
#include "set_edge_values.hpp"
#include "openmp_impl/Mesh.hpp"

namespace schro_omp
{
    template <std::floating_point real>
    void sum_side(std::vector<matrix<real>>& a, const Edge& edge, const Mesh<real>& mesh)
    {
        std::vector<real> edge_vals[2];
        for (int k=0; k < 2; ++k)
        { 
            int e = edge.elements[k];
            int s = std::abs(edge.element_sides[k]);

            edge_vals[k] = get_edge_values<real>(a, mesh, e, s);
        }

        int n = edge.startIter;
        int dn = edge.deltaIter;

        for (int j=1; j < mesh.N-1; ++j)
        {
            real edgeSum = edge_vals[0][j] + edge_vals[1][n];
            edge_vals[0][j] = edgeSum;
            edge_vals[1][n] = edgeSum;
            n += dn;
        } // compute edge contribution from both elements

        for (int k=0; k < 2; ++k)
        {
            int e = edge.elements[k];
            int s = std::abs(edge.element_sides[k]);

            if (s == 2 or s == 4) {
                int i = mesh.smap(s);
                for (int j=1; j < mesh.N-1; ++j)
                    a[e].at(i, j) = edge_vals[k][j];
            } else {
                int j = mesh.smap(s);
                for (int i=1; i < mesh.N-1; ++i)
                    a[e].at(i, j) = edge_vals[k][i];
            }
        } // distribute edge contribution to the local information of both elements.
    }

    template <std::floating_point real>
    void sum_node(std::vector<matrix<real>>& a, const CornerNode<real>& node, const Mesh<real>& mesh)
    {
        real csum = 0;
        
        for (auto el = node.connected_elements.cbegin(); el != node.connected_elements.cend(); ++el)
            csum += a[el->element_id].at(el->i, el->j);
        
        for (auto el = node.connected_elements.cbegin(); el != node.connected_elements.cend(); ++el)
            a[el->element_id].at(el->i, el->j) = csum;
    }

    template <std::floating_point real>
    void global_sum(std::vector<matrix<real>>& a, const Mesh<real>& mesh)
    {
        #pragma omp parallel for if(mesh.edges.size() > 63) schedule(static) 
        for (auto it = mesh.edges.cbegin(); it != mesh.edges.cend(); ++it)
            if (it->edge_type != BOUNDARY)
                sum_side(a, *it, mesh);

        #pragma omp parallel for if(mesh.nodes.size() > 63) schedule(static) 
        for (auto it = mesh.nodes.cbegin(); it != mesh.nodes.cend(); ++it)
            if (it->corner_type != BOUNDARY)
                sum_node(a, *it, mesh);
    }
    
} // namespace schro_omp

#endif