#ifndef SOLVER_BASE_HPP
#define SOLVER_BASE_HPP

#include <concepts>

template <typename real>
struct solver_results
{
    bool success;
    int n_iter;
    real residual;
};

class IdentityPreconditioner
{
    public:
    template <typename vec>
    vec operator()(const vec& x)
    {
        return x;
    }
};

#endif