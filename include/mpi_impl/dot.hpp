#ifndef MPI_IMPL_DISTRIBUTED_DOT_HPP
#define MPI_IMPL_DISTRIBUTED_DOT_HPP

#include <unordered_map>

#include <boost/mpi.hpp>
#include <armadillo>

#include "types.hpp"
#include "get_edge_values.hpp"
#include "mpi_impl/Mesh.hpp"

namespace schro_mpi
{
    template <std::floating_point real>
    real dot(const Mesh<real>& mesh, const SparseData<matrix<real>>& a, const SparseData<matrix<real>>& b)
    {
        int n = mesh.N;

        // compute dot for interior nodes
        real p_interiors = 0;

        auto interior = arma::span(1, n-2);
        for (const auto& [el, u] : a)
        {
            const auto& v = b.at(el);
            p_interiors += arma::dot(u.submat(interior, interior), v.submat(interior, interior));
        }

        // compute dot along edges
        real p_edges = 0;

        for (const auto& [id, edge] : mesh.edges)
        {
            const int e = edge.elements[0];
            if (mesh.elements.contains(e)) {
                const int s = std::abs(edge.element_sides[0]);
                const auto& u = a.at(e);
                const auto& v = b.at(e);

                real pe = 0;
                if (s == 2 or s == 4) {
                    int i = mesh.smap(s);
                    for (int j=1; j < mesh.N-1; ++j)
                        pe += u.at(i,j) * v.at(i,j);
                } else {
                    int j = mesh.smap(s);
                    for (int i=1; i < mesh.N-1; ++i)
                        pe += u.at(i,j) * v.at(i,j);
                }

                p_edges += pe;
            }
        }

        // compute dot on corners
        real p_corners = 0;

        for (const auto& [id, node] : mesh.nodes)
        {
            const auto& info = node.connected_elements[0];
            if (mesh.elements.contains(info.element_id))
                p_corners += a.at(info.element_id).at(info.i, info.j) * b.at(info.element_id).at(info.i, info.j);
        }
        
        real p = p_interiors + p_edges + p_corners;

        p = mpi::all_reduce(mesh.comm, p, std::plus<real>{});

        return p;
    }

} // namespace schro_mpi

#endif