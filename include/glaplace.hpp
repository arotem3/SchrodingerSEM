#ifndef GLAPLACE_HPP
#define GLAPLACE_HPP

#include <concepts>
#include <vector>
#include <armadillo>

#include "gauss_lobatto.hpp"
#include "Quad.hpp"

// computes the Galerkin projection of the laplaction of u: (Lu, v) ==> (Du, Dv)
// algorithm 105 in:
// D. A. Kopriva. Implementing Spectral Methods for Partial Differential
// Equations: Algorithms for Scientists and Engineers. Scientific computation.
// Springer Netherlands, Dordrecht, 1. aufl. edition, 2009. ISBN 9048122600.
template <std::floating_point real>
arma::Mat<real> glaplace(const arma::Mat<real>& u, const Quad<real>& element, const arma::Mat<real>& D, const quad_rule<real>& quadrature)
{
    arma::Mat<real> u_xi = D * u;
    arma::Mat<real> u_eta = u * D.t();

    arma::Mat<real> F = D.t() * (arma::diagmat(quadrature.w) * (element.A % u_xi - element.B % u_eta));
    arma::Mat<real> G = ((element.C % u_eta - element.B % u_xi) * arma::diagmat(quadrature.w)) * D;

    arma::Mat<real> lap_u = arma::diagmat(quadrature.w) * F + G * arma::diagmat(quadrature.w);
    
    return lap_u;
}

#endif