#ifndef MASK_NODE_HPP
#define MASK_NODE_HPP

#include <concepts>

#include "CornerNode.hpp"

template <std::floating_point real, typename container, std::predicate<int> pred>
void mask_node(container& a, const CornerNode<real>& node, pred referencable_element)
{
    // we set all internal element contributions to zero except for the
    // first connected element unless the corner is on the boundary in which
    // case we set the first element contribution to zero as well to
    // accomodate the boundary conditions.
    auto el = node.connected_elements.cbegin();
    if (node.corner_type != BOUNDARY)
        ++el; // skip first element unless it is on the boundary
    for (; el != node.connected_elements.cend(); ++el) {
        if (referencable_element(el->element_id))
            a[el->element_id].at(el->i, el->j) = 0.0; // set redundant contributions to zero.
    }
}

#endif