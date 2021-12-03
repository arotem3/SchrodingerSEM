#ifndef MPI_IMPL_UNMASK_HPP
#define MPI_IMPL_UNMASK_HPP

#include <concepts>

#include "types.hpp"
#include "mpi_impl/comm_base.hpp"
#include "mpi_impl/Mesh.hpp"
#include "get_edge_values.hpp"
#include "set_edge_values.hpp"

namespace schro_mpi
{
    template <std::floating_point real>
    void unmask_side(SparseData<matrix<real>>& a, int edge_id, const Edge& edge, const Mesh<real>& mesh, EdgePromises<real>& promises, std::set<int>& expected_recieves, const std::unordered_map<int,int>& E2P)
    {
        int e0 = edge.elements[0]; // the element managing this edge
        int s0 = edge.element_sides[0];

        if (a.contains(e0)) {
            // this processor owns the element which manages this edge, so we can
            // unmask the elements on the other side of the edge, or promise to send
            // the edge data to the processor that does own that element.
            std::vector<real> edge_values = get_edge_values<real>(a, mesh, e0, s0);

            int e1 = edge.elements[1];
            if (a.contains(e1)) {
                // This processor owns the second element that shares this edge, so
                // we unmask the edge on this element
                int s1 = std::abs(edge.element_sides[1]);
                set_edge_values(a, mesh, edge, edge_values, e1, s1);
            } else {
                // This processor does NOT own the second element that shares this
                // edge, so we make a promise to send that owning processor the data
                // for this edge
                int partner = E2P.at(e1);
                promises[partner].push_back(EdgeMessage<real>(edge_id, std::move(edge_values)));
            }
        } else {
            // this processor does NOT own the element that manages this edge, so it
            // makes a note that it is owed data from the processor owning this
            // element.
            int partner = E2P.at(e0);
            expected_recieves.insert(partner);
        }
    }

    template <std::floating_point real>
    void unmask_node(SparseData<matrix<real>>& a, const CornerNode<real>& node, CornerPromises<real>& promises, std::set<int>& expected_recieves, const std::unordered_map<int, int>& E2P)
    {
        auto el = node.connected_elements.cbegin(); // first connected element is reference element
        const int i0 = el->i;
        const int j0 = el->j;
        const int e0 = el->element_id;
        if (a.contains(e0)) {
            // This processor owns the element that manages this corner, so
            // it will assign the value of this corner to all the other
            // elements that need it.
            const real cval = a[e0].at(i0,j0);
            ++el;
            for (; el != node.connected_elements.cend(); ++el)
            {
                if (a.contains(el->element_id))
                    // This processor owns this element, so it will assign
                    // the corner value to that element
                    a.at(el->element_id)(el->i, el->j) = cval;
                else {
                    // this processor does NOT own the element, so it will
                    // make a promise to send the processor that DOES own
                    // it, the data for this corner.
                    int partner = E2P.at(el->element_id);
                    promises[partner].push_back( CornerMessage<real>(cval, *el) );
                }
            }
        } else {
            // this processor does NOT own the element that manages this
            // corner, so it makes a note that it is owed data from this
            // processor.
            expected_recieves.insert(E2P.at(e0));
        }
    }

    // match values at element interfaces.
    // algorithm 131 in:
    // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
    // Equations: Algorithms for Scientists and Engineers. Scientific computation.
    // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
    template <std::floating_point real>
    void unmask(SparseData<matrix<real>>& a, const Mesh<real>& mesh, mpi::communicator& comm, const std::unordered_map<int, int>& E2P)
    {
        // we iterate over the edges we own or share. We unmask all edges we own.
        // For edges we share, if we own the element that controls that edge, then
        // we send the corresponding processor the edge data from that element, if
        // we do not own that element, we keep track of the processor that owes us
        // that data.

        EdgePromises<real> edge_promises;
        std::set<int> expected_recieves;
        for (const auto& [edge_id, edge] : mesh.edges)
            if (edge.edge_type != BOUNDARY)
                unmask_side<real>(a, edge_id, edge, mesh, edge_promises, expected_recieves, E2P);

        std::vector<mpi::request> requests;
        for (const auto& [partner, messages] : edge_promises)
            requests.push_back( comm.isend(partner, 0, messages) );

        EdgePromises<real> edges_recieved;
        for (const int& partner : expected_recieves)
            requests.push_back( comm.irecv(partner, 0, edges_recieved[partner]) );

        mpi::wait_all(requests.begin(), requests.end());

        // now we've recieved all the data from the other processors, so we can
        // start unmasking those edges
        for (const auto& [partner, messages] : edges_recieved)
        {
            for (const auto& mesg : messages)
            {
                const Edge& edge = mesh.edges.at(mesg.id);
                const int e1 = edge.elements[1];
                const int s1 = std::abs(edge.element_sides[1]);
                set_edge_values<real>(a, mesh, edge, mesg.values, e1, s1);
            }
        }

        // next we do the same thing for each corner node with the slight
        // distinction that each corner node may be atributed to many elements, and,
        // therefore, multiple processors
        CornerPromises<real> corner_promises;
        expected_recieves.clear(); // reuse this variable
        for (const auto& [node_id, node] : mesh.nodes)
            if (node.corner_type != BOUNDARY)
                unmask_node<real>(a, node, corner_promises, expected_recieves, E2P);

        requests.clear(); // reuse variable
        for (const auto& [partner, messages] : corner_promises)
            requests.push_back( comm.isend(partner, 0, messages) );

        CornerPromises<real> corners_recieved;
        for (const int& partner : expected_recieves)
            requests.push_back( comm.irecv(partner, 0, corners_recieved[partner]) );

        mpi::wait_all(requests.begin(), requests.end());

        for (const auto& [partner, messages] : corners_recieved)
        {
            for (const auto& msg : messages)
            {
                a.at(msg.info.element_id)(msg.info.i, msg.info.j) = msg.value; // safe index
                // a.at(msg.info.element_id).at(msg.info.i, msg.info.j) = msg.value;
            }
        }
    }

} // namespace schro_mpi


#endif