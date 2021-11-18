#ifndef MASK_SIDE_HPP
#define MASK_SIDE_HPP

#include <concepts>

#include "types.hpp"
#include "Edge.hpp"

// set all values on an edge to zero in u. referencable_element checks whether an edge is
// contained in u, this is trivially true for the openmp implementation, but not
// needs to be checked for the mpi implementation
template <typename container, typename Mesh, std::predicate<int> pred>
void mask_side(container& u, const Edge& edge, const Mesh& mesh, pred referencable_element)
{
    // sides and corners are ordered counter-clockwise:
    // x0 ----- e4 ----- x3
    // |                 | 
    // |                 |
    // e1                e3
    // |                 |
    // |                 |
    // x1 ----- e2 ----- x2
    // since our data is organized a(i,j) = a(x[i], y[j])
    // iterate over j for fixed i on sides {2,4},
    // and iterate over i for fixed j on sides {1,3}

    int e, s;
    if (edge.edge_type == BOUNDARY) {
        // set boundary edges to zero to accomodate boundary conditions.
        // this is for Dirichlet, needs to be modified for Neumann
        e = edge.elements[0];
        s = edge.element_sides[0];
    } else {
        // for internal edges we keep the contribution from the first
        // element but set the second element to zero.
        e = edge.elements[1];
        s = std::abs(edge.element_sides[1]);
    }

    if (referencable_element(e)) {
        auto& a = u.at(e); // u[e];
        if (s == 2 or s == 4) {
            int i = mesh.smap(s);
            for (int j=1; j < mesh.N-1; ++j)
                a.at(i,j) = 0;
        } else {
            int j = mesh.smap(s);
            for (int i=1; i < mesh.N-1; ++i)
                a.at(i,j) = 0;
        }
    }
}

#endif