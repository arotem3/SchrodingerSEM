#ifndef QUAD_HPP
#define QUAD_HPP

#include <concepts>
#include <vector>
#include <cmath>
#include <armadillo>

template <std::floating_point real>
struct Quad
{
    real xs[4];
    real ys[4];
    int corner_ids[4];

    arma::Mat<real> J;
    arma::Mat<real> A;
    arma::Mat<real> B;
    arma::Mat<real> C;
    u_long n;

    // algorithms 95 and 105 in:
    // D. A. Kopriva. Implementing Spectral Methods for Partial Differential
    // Equations: Algorithms for Scientists and Engineers. Scientific computation.
    // Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
    void compute_metrics(const arma::Col<real>& z)
    {
        n = z.size();
        J.set_size(n, n);
        A.set_size(n, n);
        B.set_size(n, n);
        C.set_size(n, n);
        arma::Col<real> Xxi(n);
        arma::Col<real> Yxi(n);
        arma::Col<real> Xeta(n);
        arma::Col<real> Yeta(n);

        for (u_long i=0; i < n; ++i)
        {
            Xxi[i] = 0.25 * ((1-z[i])*(xs[1] - xs[0]) + (1+z[i])*(xs[2] - xs[3]));
            Yxi[i] = 0.25 * ((1-z[i])*(ys[1] - ys[0]) + (1+z[i])*(ys[2] - ys[3]));
            Xeta[i] = 0.25 * ((1-z[i])*(xs[3] - xs[0]) + (1+z[i])*(xs[2] - xs[1]));
            Yeta[i] = 0.25 * ((1-z[i])*(ys[3] - ys[0]) + (1+z[i])*(ys[2] - ys[1]));
        }

        for (u_long i=0; i < n; ++i)
            for (u_long j=0; j < n; ++j)
            {
                J.at(i, j) = Xxi[j]*Yeta[i] - Xeta[i]*Yxi[j];
                A.at(i, j) = ( std::pow(Yeta[i], 2) + std::pow(Xeta[i], 2) ) / J.at(i, j);
                B.at(i, j) = ( Yeta[i]*Yxi[j] + Xeta[i]*Yxi[j] ) / J.at(i, j);
                C.at(i, j) = ( std::pow(Yeta[i], 2) + std::pow(Xxi[j], 2) ) / J.at(i, j);
            }
    }

    inline std::pair<double, double> from_local_coo(double xi, double eta) const
    {
        double x = 0.25 * (xs[0]*(1-xi)*(1-eta) + xs[1]*(1+xi)*(1-eta) + xs[2]*(1+xi)*(1+eta) + xs[3]*(1-xi)*(1+eta));
        double y = 0.25 * (ys[0]*(1-xi)*(1-eta) + ys[1]*(1+xi)*(1-eta) + ys[2]*(1+xi)*(1+eta) + ys[3]*(1-xi)*(1+eta));
        return std::make_pair(x, y);
    }
};


#endif