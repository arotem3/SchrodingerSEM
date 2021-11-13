#ifndef PCG_HPP
#define PCG_HPP

#include <cmath>
#include <concepts>

#include "solver_base.hpp"

// preconditioned, first iteration
template <std::floating_point real, typename vec, std::invocable<vec, vec> Dot, std::invocable<vec> Precond>
real conj_grad(vec& p, const vec& r, Dot dot, Precond precond)
{
    p = precond(r);
    return dot(r, p);
}

// no preconditioner, first iteration
template <std::floating_point real, typename vec, std::invocable<vec, vec> Dot>
real conj_grad(vec& p, const vec& r, Dot dot, IdentityPreconditioner precond)
{
    p = r;
    return dot(r, r);
}

// preconditioned, i > 0
template <std::floating_point real, typename vec, std::invocable<vec, vec> Dot, std::invocable<vec> Precond>
real conj_grad(vec& p, const vec& r, Dot dot, Precond precond, real rho_prev)
{
    vec z = precond(r);
    real rho = dot(r, z);
    
    real beta = rho / rho_prev;
    p = z + beta * p;

    return rho;
}

// no preconditioner, i > 0
template <std::floating_point real, typename vec, std::invocable<vec, vec> Dot>
real conj_grad(vec& p, const vec& r, Dot dot, IdentityPreconditioner precond, real rho_prev)
{
    real rho = dot(r, r);
    
    real beta = rho / rho_prev;
    p = r + beta * p;

    return rho;
}

// preconditioned conjugate gradient method. We require that vec is like a
// mathematical vector, i.e. addition, substraction, and scalar multiplication
// are well defined via the operators {+, +=, -, -=, *} and the results are
// convertible to vec.
// see:
// Saad, Y. (2003). Iterative methods for sparse linear systems. Philadelphia:
// SIAM.
template <std::floating_point real, typename vec, std::invocable<vec> LinOp, std::invocable<vec, vec> Dot, std::invocable<vec> Precond>
solver_results<real> pcg(vec& x, LinOp A, const vec& b, Dot dot, Precond precond, int max_iter, real tol)
{
    vec r = b - A(x);
    real bnorm = std::sqrt(dot(b,b));

    bool success = false;

    vec p;
    real rho_prev, rho;
    int i;
    for (i=0; i < max_iter; ++i)
    {
        if (i == 0)
            rho = conj_grad<real, vec>(p, r, std::forward<Dot>(dot), std::forward<Precond>(precond));
        else
            rho = conj_grad<real, vec>(p, r, std::forward<Dot>(dot), std::forward<Precond>(precond), rho_prev);
        
        vec Ap = A(p);
        real alpha = rho / dot(p, Ap);
        x += alpha * p;
        r -= alpha * Ap;

        rho_prev = rho;

        if (std::sqrt(rho) < tol*bnorm)
        {
            success = true;
            break;
        }
    }

    solver_results<real> rslts;
    rslts.success = success;
    rslts.residual = std::sqrt(rho);
    rslts.n_iter = i;

    return rslts;
}



#endif