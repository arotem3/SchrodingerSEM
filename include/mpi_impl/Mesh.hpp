#ifndef MPI_IMPL_MESH_HPP
#define MPI_IMPL_MESH_HPP

#include <unordered_map>
#include <concepts>

#include "gauss_lobatto.hpp"
#include "Quad.hpp"
#include "Edge.hpp"
#include "CornerNode.hpp"
#include "types.hpp"
#include "mpi_impl/comm_base.hpp"

namespace schro_mpi
{
    // sparse mesh. This is used for the distributed MPI implementation where most
    // of the mesh is not stored on any one processor.
    template <std::floating_point real>
    class Mesh
    {
    friend Mesh<real> load_mesh<Mesh<real>>(int, const std::string&);
    friend std::pair<std::unordered_map<int,int>, Mesh<real>> scatter_mesh<real>(int, const std::string&, mpi::communicator&, const int);

    private:
        int _n;

    public:
        typedef real real_t;
        static const char mesh_t = 's';

        quad_rule<real> quadrature;
        matrix<real> D; // derivative operator
        SparseData<Edge> edges;
        SparseData<CornerNode<real>> nodes;
        SparseData<Quad<real>> elements;
        const int& N = _n;

        Mesh()
        {
            _n = 0;
        }
        Mesh(int p) : quadrature(gauss_lobatto<real>(p))
        {
            _n = quadrature.x.size();
        }
        
        Mesh(const quad_rule<real>& q) : quadrature(q)
        {
            _n = q.x.size();
        }
        Mesh(quad_rule<real>&& q) : quadrature(std::move(q))
        {
            _n = q.x.size();
        }


        Mesh(const Mesh<real>& mesh) : _n(mesh.N), quadrature(mesh.quadrature), D(mesh.D), edges(mesh.edges), nodes(mesh.nodes), elements(mesh.elements) {}
        Mesh<real>& operator=(const Mesh<real>& mesh)
        {
            _n = mesh.N;
            quadrature = mesh.quadrature;
            D = mesh.D;
            edges = mesh.edges;
            nodes = mesh.nodes;
            elements = mesh.elements;

            return *this;
        }

        Mesh(Mesh<real>&& mesh) : _n(mesh.N), quadrature(std::move(mesh.quadrature)), D(std::move(mesh.D)), edges(std::move(mesh.edges)), nodes(std::move(mesh.nodes)), elements(std::move(mesh.elements)) {}
        Mesh<real>& operator=(Mesh<real>&& mesh)
        {
            _n = mesh.N;
            quadrature = std::move(mesh.quadrature);
            D = std::move(mesh.D);
            edges = std::move(mesh.edges);
            nodes = std::move(mesh.nodes);
            elements = std::move(mesh.elements);

            return *this;
        }

        void compute_metrics()
        {
            for (auto& element : elements)
                element.second.compute_metrics(quadrature.x);
        }

        // maps side to index
        inline int smap(int side) const
        {
            const int smap[] = {0, N-1, N-1, 0};
            return smap[side-1];
        }

        inline int dof() const
        {
            return nodes.size() + edges.size()*(N-2) + elements.size()*(N-2)*(N-2);
        }
    };

} // namespace schro_mpi

#endif