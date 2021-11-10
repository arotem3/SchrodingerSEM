#ifndef DERIVATIVE_MATRIX_HPP
#define DERIVATIVE_MATRIX_HPP

#include <armadillo>
#include <vector>

template <typename real>
std::vector<real> barycentric_weights(const std::vector<real>& x)
{
    u_long n = x.size();

    std::vector<real> w(n, 0);
    for (u_long i=0; i < n; ++i)
    {
        w[i] = 1;
        for (u_long j=0; j < n; ++j)
            if (i != j)
                w[i] *= x[i] - x[j];
        w[i] = 1 / w[i];
    }

    return w;
}

template <typename real>
arma::Mat<real> derivative_matrix(const std::vector<real>& x)
{
    auto w = barycentric_weights<real>(x);

    u_long n = x.size();
    arma::Mat<real> D(n, n, arma::fill::zeros);

    for (u_long i=0; i < n; ++i)
    {
        D.at(i, i) = 0;
        for (u_long j=0; j < n; ++j)
            if (j != i)
            {
                D.at(i, j) = (w[j] / w[i]) / (x[i] - x[j]);
                D.at(i, i) -= D.at(i, j);
            }
    }

    return D;
}

#endif