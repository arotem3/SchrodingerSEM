#ifndef EDGE_HPP
#define EDGE_HPP

#include "MeshBase.hpp"

class Edge
{
public:
    CornerEdgeType edge_type;
    int elements[2]; // elements sharing this edge
    int element_sides[2]; // the sides of those elements, one of +/-{0,1,2,3,4}
    int startIter;
    int deltaIter;

    Edge () {}
    
    // initialize edge by specifying the key to the element that controls it and
    // the side of the element that it is one
    explicit Edge(int el, int side)
    {
        edge_type = BOUNDARY;
        elements[0] = el;
        elements[1] = -1; // -1 --> place holder for nothing
        element_sides[0] = side;
        element_sides[1] = 0;
    }
};

#endif