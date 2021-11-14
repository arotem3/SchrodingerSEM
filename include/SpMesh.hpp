#ifndef SPMESH_HPP
#define SPMESH_HPP

#include <unordered_map>
#include <concepts>

#include <armadillo>

#include "gauss_lobatto.hpp"
#include "Quad.hpp"
#include "Edge.hpp"
#include "CornerNode.hpp"

// sparse mesh. This is used for the distributed MPI implementation where most
// of the mesh is not stored on any one processor.
template <std::floating_point real>
class SpMesh
{
friend SpMesh<real> load_mesh<SpMesh<real>>(int, const std::string&);

private:
    int _n;

public:
    typedef real real_t;

    quad_rule<real> quadrature;
    matrix<real> D; // derivative operator
    SparseData<Edge> edges;
    SparseData<CornerNode<real>> nodes;
    SparseData<Quad<real>> elements;
    const int& N = _n;

    // maps side to index
    int smap(int side) const
    {
        static const int smap[] = {0, N-1, N-1, 0};
        return smap[side-1];
    }

    int dof() const
    {
        return nodes.size() + edges.size()*(N-2) + elements.size()*(N-2)*(N-2);
    }
};

#endif