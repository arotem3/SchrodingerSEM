#ifndef GET_EDGE_VALUES_HPP
#define GET_EDGE_VALUES_HPP

#include <concepts>
#include <vector>

template <std::floating_point real, typename container, typename Mesh>
std::vector<real> get_edge_values(container& a, const Mesh& mesh, int e, int s)
{
    std::vector<real> edge_vals(mesh.N, 0.0);
    if (s == 2 or s == 4) {
        int i = mesh.smap(s);
        for (int j=1; j < mesh.N-1; ++j)
            edge_vals[j] = a[e].at(i, j);
    } else {
        int j = mesh.smap(s);
        for (int i=1; i < mesh.N - 1; ++i)
            edge_vals[i] = a[e].at(i, j);
    }

    return edge_vals;
}

#endif