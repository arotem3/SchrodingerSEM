#ifndef DISTRIBUTE_MESH_HPP
#define DISTRIBUTE_MESH_HPP

#include <unordered_map>
#include <vector>
#include <queue>
#include <concepts>

#include "SpMesh.hpp"

template <std::floating_point real>
struct _idC
{
    int id;
    real x;
    real y;
    real pc1;
};

class _idcLess
{
public:
    template <std::floating_point real>
    bool operator()(const _idC<real>& a, const _idC<real>& b) const
    {
        return a.pc1 < b.pc1;
    }
};

template <std::floating_point real>
void PCA1(std::vector<_idC<real>>& a, int first, int last)
{
    real xm = 0.0;
    real ym = 0.0;
    real xv = 0.0;
    real yv = 0.0;
    real cov = 0.0;

    const int n = last - first;

    for (int i=first; i < last; ++i)
    {
        xm += a[i].x;
        ym += a[i].y;
    }
    xm /= n;
    ym /= n;

    for (int i=first; i < last; ++i)
    {
        const real s = (a[i].x - xm);
        const real t = (a[i].y - ym);
        xv += s*s;
        yv += t*t;
        cov += s*t;
    }
    xv /= n-1;
    yv /= n-1;
    cov /= n-1;

    const real D = std::sqrt(std::pow(xv - yv, 2) + std::pow(2*cov, 2));
    const real eigenvec_x = (D + xv - yv) / (2*cov);
    const real eigenvec_y = 1.0;

    for (int i=first; i < last; ++i)
        a[i].pc1 = eigenvec_x*(a[i].x - xm) + eigenvec_y*(a[i].y - ym);
}

// computes a P groups that split the mesh in a way that ensure elements in a
// given group are connected and uses the geometry to try and minimize the
// 'surface area' of the group. The map ouput takes an element id as input and
// returns a group id from 0 to p-1. p must be a power of 2. Splitting is
// computed recursively using a quadtree of the coordinates of the element
// centers along 'optimal directions' determined by the first principle
// component of the data.
template <std::floating_point real>
std::unordered_map<int, int> distribute_mesh(const SpMesh<real>& mesh, int P)
{
    const int n = mesh.elements.size();

    std::unordered_map<int,int> out;

    std::vector<_idC<real>> a(n);
    int i=0;
    for (const auto& [id, elem] : mesh.elements) {
        real xbar = 0.0;
        real ybar = 0.0;
        for (int i=0; i < 4; ++i) {
            xbar += elem.xs[i]*real(0.25);
            ybar += elem.ys[i]*real(0.25);
        }
        a[i].x = xbar;
        a[i].y = ybar;
        a[i].id = id;
        ++i;

        out[id] = 0;
    }

    std::queue<std::tuple<int,int,int>> q;
    int label = 1;
    q.push(std::make_tuple(0,n,P));
    while (not q.empty()) {
        auto [first, last, p] = q.front(); q.pop();
        int m = last - first;

        PCA1(a, first, last);

        int median_loc = (last + first) / 2;

        std::nth_element(a.begin()+first, a.begin()+median_loc, a.begin()+last, _idcLess());

        for (int i=median_loc; i < last; ++i)
        {
            out[a[i].id] = label;
        }
        label++;

        if (p > 2)
        {
            q.push(std::make_tuple(first, median_loc, p/2));
            q.push(std::make_tuple(median_loc, last, p/2));
        }
    }

    return out;
}

#endif