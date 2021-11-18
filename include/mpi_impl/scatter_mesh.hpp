#ifndef SCATTER_MESH_HPP
#define SCATTER_MESH_HPP

#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>

#include "types.hpp"
#include "load_mesh.hpp"
#include "mpi_impl/Mesh.hpp"
#include "mpi_impl/comm_base.hpp"
#include "mpi_impl/distribute_mesh.hpp"

namespace schro_mpi
{
    // loads a mesh from folder (on the rootProcessor) and distribute the mesh
    // to the rest of the processor in the communicator (load balanced).
    template <std::floating_point real>
    std::pair<std::unordered_map<int,int>, Mesh<real>> scatter_mesh(int p, const std::string& folder, mpi::communicator& comm, const int rootProcessor)
    {
        const int num_proc = comm.size();
        const int rank = comm.rank();
        if (rank == rootProcessor) {
            Mesh<real> super_mesh = load_mesh<Mesh<real>>(p, folder);      
            Mesh<real> mesh(std::move(super_mesh.quadrature));
            
            std::unordered_map<int, int> E2P;
            if (rank == rootProcessor)
                E2P = distribute_mesh<real>(super_mesh, num_proc);

            mpi::broadcast(comm, E2P, rootProcessor);

            std::vector< std::unordered_map<int,CornerNode<real>> > needed_corners(num_proc);
            std::vector< std::unordered_map<int,Quad<real>> >       needed_elements(num_proc);
            std::vector< std::unordered_map<int,Edge> >             needed_edges(num_proc);

            for (int i=0; i < super_mesh.elements.size(); ++i)
            {
                // determine which elements belong to which processors and place
                // them into a map of {int, Quad} for the owner of that element.
                // That map will be sent to the owner.
                int elem_owner = E2P.at(i);
                needed_elements.at(elem_owner)[i] = super_mesh.elements.at(i);
                for (int j=0; j < 4; ++j)
                {
                    // determine which corners are needed by element i and insert
                    // their id into a map of {int, CornerNode} for the owner of
                    // that element. That map will be sent to the owner once all the
                    // corners are determined.
                    int corner_id = super_mesh.elements.at(i).corner_ids[j];
                    if (not needed_corners.at(elem_owner).contains(corner_id))
                        needed_corners.at(elem_owner)[corner_id] = super_mesh.nodes[corner_id];
                }
            }

            for (int i=0; i < super_mesh.edges.size(); ++i)
            {
                // same for edges now.
                int e = super_mesh.edges.at(i).elements[0];
                int elem_owner = E2P.at(e);
                if (needed_edges[elem_owner].count(i) == 0)
                    needed_edges[elem_owner][i] = super_mesh.edges.at(i);

                if (super_mesh.edges.at(i).edge_type != BOUNDARY) {
                    e = super_mesh.edges.at(i).elements[1];
                    elem_owner = E2P.at(e);
                    if (needed_edges[elem_owner].count(i) == 0)
                        needed_edges[elem_owner][i] = super_mesh.edges.at(i);
                }
            }

            mpi::scatter(comm, needed_corners, mesh.nodes, rootProcessor);
            mpi::scatter(comm, needed_edges, mesh.edges, rootProcessor);
            mpi::scatter(comm, needed_elements, mesh.elements, rootProcessor);

            return std::make_pair(std::move(E2P), std::move(mesh));
        } else {
            Mesh<real> mesh(p);
            std::unordered_map<int, int> E2P;
            
            mpi::broadcast(comm, E2P, rootProcessor);

            mpi::scatter(comm, mesh.nodes, rootProcessor);
            mpi::scatter(comm, mesh.edges, rootProcessor);
            mpi::scatter(comm, mesh.elements, rootProcessor);

            return std::make_pair(std::move(E2P), std::move(mesh));
        }
    }

} // namespace schro_mpi

#endif