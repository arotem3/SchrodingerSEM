#ifndef CORNERNODE_HPP
#define CORNERNODE_HPP

#include <vector>

#include "MeshBase.hpp"

struct ElementCornerInfo
{
    int element_id; // index in mesh.elements
    int i; // (i,j) is the index of the corner relative to the element, e.g. u.at(element_id)[i][j] is the value at the corner
    int j;
};

template <std::floating_point real>
class CornerNode
{
public:
    CornerEdgeType corner_type;
    real x, y;
    std::vector<ElementCornerInfo> connected_elements;

    CornerNode(real xx=0.0, real yy=0.0)
    {
        x = xx;
        y = yy;
        corner_type = INTERIOR;
    }
};

#endif