#ifndef MPI_IMPL_GLOBAL_SUM_HPP
#define MPI_IMPL_GLOBAL_SUM_HPP

#include <concepts>

#include "types.hpp"
#include "get_edge_values.hpp"
#include "set_edge_values.hpp"
#include "mpi_impl/comm_base.hpp"
#include "mpi_impl/Mesh.hpp"

namespace schro_mpi
{
    template <std::floating_point real>
    void sum_side(SparseData<matrix<real>>& a, int edge_id, const Edge& edge, const Mesh<real>& mesh, EdgePromises<real>& promises, std::set<int>& expected_recieves, const std::unordered_map<int,int>& E2P)
    {
        if (a.contains(edge.elements[0]) and a.contains(edge.elements[1])) {
            // this proc owns both elements on either side of this edge, it can
            // compute the edge sum directly.
            // std::vector<std::vector<real>> edge_vals = {std::vector<real>(mesh.N, 0), std::vector<real>(mesh.N, 0)};
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
        } else {
            int k = a.contains(edge.elements[0]) ? 0 : 1;

            int partner = E2P.at(edge.elements[1-k]);
            
            int e = edge.elements[k];
            int s = std::abs(edge.element_sides[k]);
        
            EdgeMessage<real> msg;
            msg.id = edge_id;
            msg.values = get_edge_values<real>(a, mesh, e, s);

            promises[partner].push_back(std::move(msg));
            expected_recieves.insert(partner);
        }
    }

    // sum contributions at element interfaces
    // algorithm 132:
    // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
    // Equations: Algorithms for Scientists and Engineers. Scientific computation.
    // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
    template <std::floating_point real>
    void global_sum(SparseData<matrix<real>>& a, const Mesh<real>& mesh, mpi::communicator& comm, const std::unordered_map<int, int>& E2P)
    {
        // First we compute sums along edges. When this processor owns both elements
        // making up an edge, it simply sum those edges. If this proc owns only one
        // of the elements but not the other, this proc send the proc owning the
        // other element the data for this proc's edge and it expects the same data
        // in return. Each processor in this exchange computes its own sum from the
        // exchanged data (even though it is the same value, we eliminate the need
        // for further communication)
        EdgePromises<real> edge_promises;
        std::set<int> expected_recieves;
        for (const auto& [edge_id, edge] : mesh.edges)
            if (edge.edge_type != BOUNDARY)
                sum_side(a, edge_id, edge, mesh, edge_promises, expected_recieves, E2P);

        std::vector<mpi::request> requests;
        for (const auto& [partner, messages] : edge_promises)
            requests.push_back( comm.isend(partner, 0, messages) );

        EdgePromises<real> edges_recieved;
        for (const int& partner : expected_recieves)
            requests.push_back( comm.irecv(partner, 0, edges_recieved[partner]) );

        mpi::wait_all(requests.begin(), requests.end());

        for (const auto& [partner, messages] : edges_recieved)
        {
            // now we iterate through the messages recieved my this processor
            for (const auto& msg : messages)
            {
                const Edge& edge = mesh.edges.at(msg.id);
                
                int k = a.contains(edge.elements[0]) ? 0 : 1;

                int e = edge.elements[k];
                int s = std::abs(edge.element_sides[k]);

                std::vector<real> edge_data = get_edge_values<real>(a, mesh, e, s);

                int n = edge.startIter;
                const int dn = edge.deltaIter;
                
                for (int i=1; i < mesh.N-1; ++i) {
                    // compute the edge sum, if k==0 then we are iterating directly
                    // with i, while the recieved data iterates with n. If k==1,
                    // then it is flipped
                    if (k == 0)
                        edge_data[i] += msg.values[n];
                    else
                        edge_data[n] += msg.values[i];
                    n += dn;
                }

                auto& a_e = a.at(e);
                if (s==2 or s==4) {
                    int i = mesh.smap(s);
                    for (int j=1; j < mesh.N - 1; ++j)
                        a_e(i,j) = edge_data[j];
                }
                else {
                    int j = mesh.smap(s);
                    for (int i=1; i < mesh.N - 1; ++i)
                        a_e(i,j) = edge_data[i];
                }
            }
        }

        std::unordered_map<int, real> corner_sum;
        CornerSumPromises<real> corner_promises; // proc# -> {node#, node value}
        expected_recieves.clear();
        for (const auto& [node_id, node] : mesh.nodes)
        {
            if (node.corner_type != BOUNDARY) {
                // iterate through the elements connected to this node. First
                // connected element is the owner, and manages the corner sum.
                auto el = node.connected_elements.cbegin();
                
                int owner = E2P.at(el->element_id);
                if (owner == comm.rank()) {
                    // this processor is the manager of this corner node.
                    real& csum = corner_sum[node_id];
                    csum = a[el->element_id](el->i, el->j);

                    ++el;
                    for (; el != node.connected_elements.cend(); ++el)
                    {
                        if (a.count(el->element_id) > 0) {
                            // this proc owns this element, so it adds its value to
                            // the sum
                            csum += a.at(el->element_id).at(el->i, el->j);
                        }
                        else {
                            // this proc does NOT own this element, but it expects
                            // data from it, so it makes a note to expect data from
                            // its owner
                            expected_recieves.insert(E2P.at(el->element_id));
                        }
                    }
                } else {
                    // this processor is NOT the manager of this corner node, so it
                    // sends the owner the data from the elements on this corner
                    // that it DOES own.
                    ++el;
                    for (; el != node.connected_elements.end(); ++el)
                    {
                        if (a.contains(el->element_id)) {
                            // this proc owns the current element but not the owner
                            // of the corner node, so it makes a promise to send its
                            // data to the owner
                            corner_promises[owner].push_back( std::make_pair( node_id, a.at(el->element_id).at(el->i, el->j) ) );
                        }
                    }
                }
            }
        }

        requests.clear(); // reuse variable
        // send corner data to owners
        for (const auto& [partner, messages] : corner_promises)
            requests.push_back( comm.isend(partner, 0, messages) );

        CornerSumPromises<real> corners_recieved;
        // recieve corner data from nodes sharing corner
        for (const int& partner : expected_recieves)
            requests.push_back( comm.irecv(partner, 0, corners_recieved[partner]) );

        mpi::wait_all(requests.begin(), requests.end());

        // compute corner sums
        for (const auto& [partner, messages] : corners_recieved)
            for (const auto& [node_id, cval] : messages)
                corner_sum.at(node_id) += cval;

        CornerSumPromises<real> sums_bcast;
        // need to broad-cast the sum from the manager of the corner-node to the
        // other elements sharing the corner node. We use the information from the
        // recieved corners to determine which corners need this proc's data
        for (const auto& [partner, messages] : corners_recieved)
            for (const auto& [node_id, val] : messages)
                sums_bcast[partner].push_back( std::make_pair(node_id, corner_sum.at(node_id)) );

        requests.clear(); // reuse variable
        // broadcast corner sums to all who share corner
        for (const auto& [partner, messages] : sums_bcast)
            requests.push_back( comm.isend(partner, 0, messages) );

        // recieve corner sums from owners. We use the information from the sent
        // corners to determine which procs will send this proc data
        corners_recieved.clear(); // reuse variables
        for (const auto& [partner, messages] : corner_promises)
            requests.push_back( comm.irecv(partner, 0, corners_recieved[partner]) );

        mpi::wait_all(requests.begin(), requests.end());

        // assign corner sums to corresponding location from sums this proc managed
        for (const auto& [node_id, csum] : corner_sum)
            for (const auto& el : mesh.nodes.at(node_id).connected_elements)
                if (a.contains(el.element_id))
                    a.at(el.element_id).at(el.i, el.j) = csum;

        // assign corner sums to corresponding location from sums recieved
        for (const auto& [partner, messages] : corners_recieved)
            for (const auto& [node_id, csum] : messages)
                for (const auto& el : mesh.nodes.at(node_id).connected_elements)
                    if (a.contains(el.element_id))
                        a.at(el.element_id).at(el.i, el.j) = csum;
    }

} // namespace schro_mpi

#endif