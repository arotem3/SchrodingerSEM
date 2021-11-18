#ifndef MPI_IMPL_DISTRIBUTED_DOT_HPP
#define MPI_IMPL_DISTRIBUTED_DOT_HPP

#include <unordered_map>

#include <boost/mpi.hpp>
#include <armadillo>

#include "types.hpp"
#include "mpi_impl/Mesh.hpp"

namespace schro_mpi
{
    namespace mpi = boost::mpi;

    template <std::floating_point real>
    real dot(mpi::communicator& comm, const Mesh<real>& mesh, const SparseData<matrix<real>>& a, const SparseData<matrix<real>>& b)
    {
        int n = mesh.N;

        // compute dot for interior nodes
        real p_interiors = 0;

        for (int i=0; i < a.size(); ++i)
            p_interiors += arma::dot(a.at(i).submat(arma::span(1,n-1), arma::span(1,n-1)), b.at(i).submat(arma::span(1,n-1), arma::span(1,n-1)));

        // compute dot along edges
        real p_edges = 0;

        for (int i=0; i < mesh.edges.size(); ++i)
        {
            auto& edge = mesh.edges.at(i);
            int e = edge.elements[0];
            if (mesh.elements.contains(e)) {
                auto& v = b.at(e);
                auto& u = a.at(e);
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
        }

        // compute dot on corners
        real p_corners = 0;
        
        for (int i=0; i < mesh.nodes.size(); ++i)
        {
            auto& node = mesh.nodes.at(i).connected_elements[0];
            if (mesh.elements.contains(node.element_id))
                p_corners += a.at(node.element_id).at(node.i, node.j) * b.at(node.element_id).at(node.i, node.j);
        }

        real p = p_interiors + p_edges + p_corners;

        mpi::all_reduce(comm, p, std::plus<real>{});

        return p;
    }

} // namespace schro_mpi

#endif