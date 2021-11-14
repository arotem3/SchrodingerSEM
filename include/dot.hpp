#ifndef DOT_HPP
#define DOT_HPP

#include "Mesh.hpp"
#include "solution_wrapper.hpp"

// computes dot product between two mesh-functions, for use with linear solvers.
template <std::floating_point real>
real dot(const Mesh<real>& mesh, const std::vector<arma::Mat<real>>& a, const std::vector<arma::Mat<real>>& b)
{
    int n = mesh.N;

    // compute dot for interior nodes
    real p_interiors = 0;

    #pragma omp parallel for reduction(+:p_interior) schedule(dynamic) nowait
    for (int i=0; i < a.size(); ++i)
        p_interiors += arma::dot(a[i].submat(arma::span(1,n-1), arma::span(1,n-1)), b[i].submat(arma::span(1,n-1), arma::span(1,n-1)));

    // compute dot along edges
    real p_edges = 0;

    #pragma omp parallel for reduction(+:p_edges) schedule(dynamic) nowait
    for (int i=0; i < mesh.edges.size(); ++i)
    {
        auto& edge = mesh.edges[i];
        auto& u = a.at(edge.elements[0]);
        auto& v = b.at(edge.elements[0]);
        int s = std::abs(edge.element_sides[0]);
        
        real pe = 0;
        
        if (s == 2 or s == 4) {
            int i = mesh.smap(s);
            for (int j=1; j < n-1; ++j)
                pe += u(i, j) * v(i, j);
        } else {
            int j = mesh.smap(s);
            for (int i=1; i < n-1; ++i)
                pe += u(i, j) * v(i, j);
        }

        p_edges += pe;
    }

    // compute dot on corners
    real p_corners = 0;
    
    #pragma omp parallel for reduction(+:p_corners) schedule(dynamic) nowait
    for (int i=0; i < mesh.nodes.size(); ++i)
    {
        auto& node = mesh.nodes[i].connected_elements[0];
        p_corners += a.at(node.element_id)(node.i, node.j) * b.at(node.element_id)(node.i, node.j);
    }

    return p_interiors + p_edges + p_corners;
}

#endif