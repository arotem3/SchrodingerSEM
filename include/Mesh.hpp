#ifndef MESH_HPP
#define MESH_HPP

#include <vector>
#include <concepts>

#include <armadillo>

#include "gauss_lobatto.hpp"
#include "Quad.hpp"
#include "Edge.hpp"
#include "CornerNode.hpp"

template <std::floating_point real>
class Mesh
{
friend Mesh<real> load_mesh<Mesh<real>>(int, const std::string&);

private:
    int _n;

public:
    typedef real real_t;

    quad_rule<real> quadrature;
    matrix<real> D; // derivative operator
    std::vector<Edge> edges;
    std::vector<CornerNode<real>> nodes;
    std::vector<Quad<real>> elements;
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