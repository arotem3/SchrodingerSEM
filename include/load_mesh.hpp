#ifndef LOAD_MESH_HPP
#define LOAD_MESH_HPP

#include <utility>
#include <string>
#include <cmath>
#include <unordered_map>
#include <concepts>

#include "Edge.hpp"
#include "CornerNode.hpp"
#include "gauss_lobatto.hpp"
#include "derivative_matrix.hpp"
#include "Mesh.hpp"
#include "SpMesh.hpp"

// Cantor's pairing function
// this is used to index the sparse data structures
inline int hash_pair(const int& x, const int& y)
{
    return (x + y) * (x + y + 1) / 2 + y;
}

// inverse of Cantor's pairing function
inline std::pair<int,int> inverse_hash_pair(const int& z)
{
    const int w = (std::sqrt(8*z + 1) - 1) / 2;
    const int t = w*(w + 1) / 2;
    const int y = z - t;
    return std::make_pair(w - y, y);
}

class EdgeMap
{
private:
    std::unordered_map<int, Edge> _x;

public:
    inline Edge& operator[](const std::pair<int,int>& key)
    {
        return _x[hash_pair(key.first, key.second)];
    }

    inline std::unordered_map<int,Edge>::iterator begin()
    {
        return _x.begin();
    }

    inline std::unordered_map<int,Edge>::iterator end()
    {
        return _x.end();
    }

    inline bool contains(const std::pair<int,int>& key) const
    {
        return _x.contains(hash_pair(key.first, key.second));
    }

    inline u_long size() const
    {
        return _x.size();
    }
};


// loads a mesh from folder with three files: 'info.txt' (has two numbers, # of
// nodes in mesh, # of elements); 'coordinates.txt' (each line has two numbers
// identifying the x and y coordinates of the nodes of the mesh); 'elements.txt'
// (has 4 integers on each line identifying the index of the corners making up
// the element with respect to the line number in 'coordinates.txt' starting at
// 0, the coordinates should be ordered counter clockwise). The parameter p is
// polynomial order on the elements.
// see chapter 8 section 2 of:
// D. A. Kopriva. Implementing Spectral Methods for Partial Differential
// Equations: Algorithms for Scientists and Engineers. Scientific computation.
// Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
template <typename MeshT>
MeshT load_mesh(int p, const std::string& folder)
{
    typedef typename MeshT::real_t real;

    static const int cmap1[] = {0, p-1, p-1, 0};
    static const int cmap2[] = {0, 0, p-1, p-1};
    static const int emap1[] = {0,1,3,0};
    static const int emap2[] = {1,2,2,3};

    // initialize quadrature rule and derivative operator
    MeshT mesh;
    mesh.quadrature = gauss_lobatto<real>(p);
    mesh._n = mesh.quadrature.x.size();
    mesh.D = derivative_matrix<real>(mesh.quadrature.x);

    // read mesh parameters
    std::ifstream info_file(folder + "/info.txt");
    if (not info_file.is_open())
        throw std::runtime_error("could not open info file.");
    
    int n_nodes, n_el;
    info_file >> n_nodes >> n_el;
    info_file.close();

    // load node coordinates
    std::ifstream coo_file(folder + "/coordinates.txt");
    if (not coo_file.is_open())
        throw std::runtime_error("could not open coordinate file.");
    
    if constexpr(std::is_same<MeshT, Mesh<real>>::value)
        mesh.nodes.resize(n_nodes);

    for (int k=0; k < n_nodes; ++k)
    {
        real x, y;
        coo_file >> x >> y;
        mesh.nodes[k] = CornerNode<real>(x,y);
    }
    coo_file.close();

    // construct elements from corners.
    std::ifstream element_file(folder + "/elements.txt");
    if (not element_file.is_open())
        throw std::runtime_error("could not open element file.");

    if constexpr(std::is_same<MeshT, Mesh<real>>::value)
        mesh.elements.resize(n_el);

    for (int el=0; el < n_el; ++el)
    {
        Quad<real> element;
        for (int i=0; i < 4; ++i)
        {
            int c;
            element_file >> c;

            ElementCornerInfo info;
            info.element_id = el;
            info.i = cmap1[i];
            info.j = cmap2[i];
            mesh.nodes.at(c).connected_elements.push_back(info);
            
            element.corner_ids[i] = c;
            element.xs[i] = mesh.nodes[c].x;
            element.ys[i] = mesh.nodes[c].y;
        }
        element.compute_metrics(mesh.quadrature.x);
        mesh.elements[el] = std::move(element);
    }
    element_file.close();

    // construct edges
    EdgeMap edge_map; // set of unique edges. As we search the grid for all edges, we avoid repetition
    for (int e = 0; e < n_el; ++e)
    {
        const Quad<real>& elem = mesh.elements.at(e);
        for (int i=0; i < 4; ++i)
        {
            int l1 = emap1[i];
            int l2 = emap2[i];
            int start = elem.corner_ids[l1]; // edge : start <-> end
            int end = elem.corner_ids[l2];

            std::pair<int,int> key = std::make_pair(std::min(start, end), std::max(start,end)); // key is sorted to ensure uniqueness

            if (edge_map.contains(key))
            {
                int e0 = edge_map[key].elements[0];
                int s0 = edge_map[key].element_sides[0];
                int n1 = mesh.elements.at(e0).corner_ids[emap1[s0-1]];

                edge_map[key].elements[1] = e;
                edge_map[key].edge_type = INTERIOR;
                if (start == n1)
                    edge_map[key].element_sides[1] = i+1;
                else
                    edge_map[key].element_sides[1] = -(i+1);
            }
            else
                edge_map[key] = Edge(e, i+1);
        }
    }

    if constexpr(std::is_same<MeshT, Mesh<real>>::value)
        mesh.edges.resize(edge_map.size());

    // identify boundary edges and link nodes to edges
    int edge_i = 0;
    for (auto it=edge_map.begin(); it != edge_map.end(); ++it)
    {
        if (it->second.edge_type == BOUNDARY)
        {
            auto [n0, n1] = inverse_hash_pair(it->first);
            mesh.nodes.at(n0).corner_type = BOUNDARY;
            mesh.nodes.at(n1).corner_type = BOUNDARY;
        }
        else
        {
            if (it->second.element_sides[1] > 0)
            {
                it->second.startIter = 1;
                it->second.deltaIter = 1;
            }
            else
            {
                it->second.startIter = p-2;
                it->second.deltaIter = -1;
            }
        }
        mesh.edges[edge_i] = std::move(it->second);
        ++edge_i;
    }

    return mesh;
}

#endif