#ifndef DERIVATIVE_MATRIX_HPP
#define DERIVATIVE_MATRIX_HPP

#include "types.hpp"

// computes the derivative operator for a set of points x based on Lagrange
// polynomial interpolation on those points.
// algorithm 37 in:
// D. A. Kopriva. Implementing Spectral Methods for Partial Differential
// Equations: Algorithms for Scientists and Engineers. Scientific computation.
// Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
template <typename real>
matrix<real> derivative_matrix(const mvec<real>& x)
{
    u_long n = x.size();

    // computes the barycentric weights of the interpolating polynomials.
    mvec<real> w(n, arma::fill::ones);
    for (u_long i=0; i < n; ++i)
    {
        for (u_long j=0; j < n; ++j)
            if (i != j)
                w[i] *= x[i] - x[j];
        w[i] = 1 / w[i];
    }
    
    matrix<real> D(n, n, arma::fill::zeros);

    for (u_long i=0; i < n; ++i)
        for (u_long j=0; j < n; ++j)
            if (j != i)
            {
                D.at(i, j) = (w[j] / w[i]) / (x[i] - x[j]);
                D.at(i, i) -= D.at(i, j);
            }

    return D;
}

#endif