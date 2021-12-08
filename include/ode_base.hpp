#ifndef ODE_BASE_HPP
#define ODE_BASE_HPP

#include <concepts>

template <std::floating_point real>
struct solver_opts
{
    int max_iter;
    real tol;
    real dt;
    bool verbose;
};

class NullCallback
{
public:
    template <typename ... Args>
    constexpr bool operator()(const Args& ... args)
    {
        return false;
    }
};

#endif