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
matrix<real> glaplace(const matrix<real>& u, const Quad<real>& element, const matrix<real>& D, const quad_rule<real>& quadrature)
{
    matrix<real> u_xi = D * u;
    matrix<real> u_eta = u * D.t();

    matrix<real> F = D.t() * (arma::diagmat(quadrature.w) * (element.A % u_xi - element.B % u_eta));
    matrix<real> G = ((element.C % u_eta - element.B % u_xi) * arma::diagmat(quadrature.w)) * D;

    matrix<real> lap_u = arma::diagmat(quadrature.w) * F + G * arma::diagmat(quadrature.w);
    
    return lap_u;
}

#endif