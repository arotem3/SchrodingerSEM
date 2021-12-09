#ifndef OMP_IMPL_MESH_HPP
#define OMP_IMPL_MESH_HPP

#include <vector>
#include <concepts>

#include "types.hpp"
#include "gauss_lobatto.hpp"
#include "Quad.hpp"
#include "Edge.hpp"
#include "CornerNode.hpp"

namespace schro_omp
{

    template <std::floating_point real>
    class Mesh
    {
    friend Mesh<real> load_mesh<Mesh<real>>(int, const std::string&);

    private:
        int _n;

    public:
        typedef real real_t;
        static const char mesh_t = 'd';

        quad_rule<real> quadrature;
        matrix<real> D; // derivative operator
        std::vector<Edge> edges;
        std::vector<CornerNode<real>> nodes;
        std::vector<Quad<real>> elements;
        const int& N = _n;

        Mesh()
        {
            _n = 0;
        }
        Mesh(int p) : quadrature(gauss_lobatto<real>(p))
        {
            _n = quadrature.x.size();
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
            #pragma omp parallel for if(elements.size() > 63) schedule(static)
            for (auto it = elements.begin(); it != elements.end(); ++it)
                it->compute_metrics(quadrature.x);
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

} // namespace scho_openmp

#endif