#ifndef OMP_IMPL_DOT_HPP
#define OMP_IMPL_DOT_HPP

#include "types.hpp"
#include "openmp_impl/Mesh.hpp"
#include "solution_wrapper.hpp"

namespace schro_omp
{
    // computes dot product between two mesh-functions, for use with linear solvers.
    template <std::floating_point real>
    real dot(const Mesh<real>& mesh, const std::vector<matrix<real>>& a, const std::vector<matrix<real>>& b)
    {
        int n = mesh.N;

        // compute dot for interior nodes
        real p_interiors = 0;

        #pragma omp parallel for if(a.size() > 63) reduction(+:p_interiors) schedule(dynamic) 
        for (int i=0; i < a.size(); ++i)
            p_interiors += arma::dot(a[i].submat(arma::span(1,n-1), arma::span(1,n-1)), b[i].submat(arma::span(1,n-1), arma::span(1,n-1)));

        // compute dot along edges
        real p_edges = 0;

        #pragma omp parallel for if(mesh.edges.size() > 63) reduction(+:p_edges) schedule(dynamic) 
        for (int i=0; i < mesh.edges.size(); ++i)
        {
            auto& edge = mesh.edges[i];
            auto& u = a[edge.elements[0]];
            auto& v = b[edge.elements[0]];
            int s = std::abs(edge.element_sides[0]);
            
            real pe = 0;
            
            if (s == 2 or s == 4) {
                int i = mesh.smap(s);
                for (int j=1; j < n-1; ++j)
                    pe += u.at(i, j) * v.at(i, j);
            } else {
                int j = mesh.smap(s);
                for (int i=1; i < n-1; ++i)
                    pe += u.at(i, j) * v.at(i, j);
            }

            p_edges += pe;
        }

        // compute dot on corners
        real p_corners = 0;
        
        #pragma omp parallel for if(mesh.nodes.size() > 63) reduction(+:p_corners) schedule(dynamic) 
        for (auto it = mesh.nodes.cbegin(); it != mesh.nodes.cend(); ++it)
        {
            auto& info = it->connected_elements[0];
            p_corners += a[info.element_id].at(info.i, info.j) * b[info.element_id].at(info.i, info.j);
        }

        return p_interiors + p_edges + p_corners;
    }
    
} // namespace schro_opemp

#endif