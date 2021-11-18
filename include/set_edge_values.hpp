#ifndef SET_EDGE_VALUES_HPP
#define SET_EDGE_VALUES_HPP

#include <concepts>
#include <vector>

#include "Edge.hpp"

template <std::floating_point real, typename container, typename Mesh>
void set_edge_values(container& a, const Mesh& mesh, const Edge& edge, const std::vector<real>& edge_values, int e, int s)
{
    if (s == 2 or s == 4) { // copy tmp into corresponding side in e1
        int i = mesh.smap(s);
        int j = edge.startIter;
        for (int k=1; k < mesh.N-1; ++k)
        {
            a[e].at(i,j) = edge_values[k];
            j += edge.deltaIter;
        }
    } else {
        int j = mesh.smap(s);
        int i = edge.startIter;
        for (int k=1; k < mesh.N-1; ++k)
        {
            a[e].at(i,j) = edge_values[k];
            i += edge.deltaIter;
        }
    }
}

#endif