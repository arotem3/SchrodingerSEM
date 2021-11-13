#ifndef MINRES_HPP
#define MINRES_HPP

#include <cmath>
#include <limits>

#include "solver_base.hpp"

// attempts to solve A(x) == b (where A(x) encodes the action of a linear
// operator) using the symmetric minimum residual method. The type vec is
// required to behave like a mathematical vector, i.e. vector addition, vector
// substraction, and scalar multiplication are well defined via the operators
// {+, +=, -, -=, *} and the results are convertible to vec.
// see:
// C. C. Paige and M. A. Saunders (1975). Solution of sparse indefinite systems
// of linear equations, SIAM J. Numerical Analysis 12, 617-629.
// this implementation is a conversion of the MATLAB code found here:
// https://web.stanford.edu/group/SOL/software/minres/
template <std::floating_point real, typename vec, std::invocable<vec> LinOp, std::invocable<vec, vec> Dot, std::invocable<vec> Precond>
solver_results<real> minres(vec& x, LinOp A, const vec& b, Dot dot, Precond precond, int maxit, real tol)
{
    bool success = false;
    
    vec r1 = b - A(x);
    vec y = precond(r1);
    vec r2 = r1;

    real machine_epsilon = std::numeric_limits<real>::epsilon();
    real beta1 = std::sqrt( dot(y, r1) );
    real beta = beta1,
         beta_old = 0.0f,
         tnorm = 0.0f,
         dbar = 0.0f,
         eps = 0.0f,
         eps_old = 0.0f,
         phibar = beta1,
         cs = -1.0f,
         sn = 0.0f;

    vec w = real(0)*x;
    vec w2 = w;

    int i;
    for (i=1; i <= maxit; ++i)
    {
        vec v = (real(1)/beta) * y;
        y = A(v);

        if (i >= 2)
            y -= (beta/beta_old) * r1;

        real alpha = dot(v, y);
        y -= (alpha/beta) * r2;

        r1 = std::move(r2);
        r2 = y;

        y = precond(r2);

        beta_old = beta;
        beta = std::sqrt( dot(y,y) );

        tnorm += alpha*alpha + beta*beta + beta_old*beta_old;

        eps_old = eps;
        real delta = cs*dbar + sn*alpha;
        real gbar = sn*dbar - cs*alpha;

        eps = sn*beta;
        dbar = -cs*beta;

        real root = std::sqrt(gbar*gbar + dbar*dbar);
        real gamma = std::max(std::sqrt(gbar*gbar + beta*beta), machine_epsilon);
        cs = gbar / gamma;
        sn = beta / gamma;

        real phi = cs * phibar;
        phibar *= sn;

        vec w1 = std::move(w2);
        w2 = std::move(w);

        w = (real(1)/gamma) * (v - eps_old*w1 - delta*w2);
        x += phi*w;

        real A_norm = std::sqrt(tnorm);
        real y_norm = std::sqrt( dot(x,x) );
        
        bool residual_convergence = phibar < tol*(A_norm * y_norm) + tol;
        bool residual_orthogonality = root < tol*A_norm + tol;
        if (residual_convergence or residual_orthogonality) {
            success = true;
            break;
        }
    }

    solver_results<real> rslts;
    rslts.success = success;
    rslts.residual = phibar;
    rslts.n_iter = i;
    return rslts;
}

#endif