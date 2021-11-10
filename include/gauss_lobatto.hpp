#ifndef GAUSS_LOBATTO_HPP
#define GAUSS_LOBATTO_HPP

#include <vector>
#include <cmath>
#include <tuple>
#include <concepts>

#include <boost/math/special_functions/legendre.hpp>

// extern to lapack routine dsteqr for eigevalue decomposition of symmetric
// tridiagonal matrix:
// COMPZ: is a single character, set to 'N' for only eigenvalues
// N: order of matrix
// D: pointer to matrix diagonal (length n)
// E: pointer to matrix off diagonal (length n-1)
// Z: pointer to orthogonal matrix, not necessary here
// LDZ_dummy: leading dim of Z, just set to 1 since it wont be used
// WORK: pointer to array, not needed for COMPZ=='N'
// INFO: 0 on success, <0 failed, >0 couldn't find all eigs
extern "C" int dsteqr_(char* COMPZ, int* N, double* D, double* E, double* Z, int* LDZ_dummy, double* WORK, int* INFO);

// like dsteqr but for single precision
extern "C" int ssteqr_(char*, int*, float*, float*, float*, int*, float*, int*);

template <std::floating_point real>
std::pair<std::vector<real>, std::vector<real>> gauss_lobatto(int n)
{
    if (n < 1)
        throw std::invalid_argument("gauss_lobatto error: require n >= 1, but n =" + std::to_string(n) + ".");
        
    static const std::unordered_map<int, std::vector<real>> lgl =
    {
        {1, {-1,1}},
        {2, {-1,0,1}},
        {3, {-1, -0.447213595499958, 0.447213595499958, 1}},
        {4, {-1, -0.654653670707977, 0, 0.654653670707977, 1}},
        {5, {-1, -0.765055323929465, -0.285231516480645, 0.285231516480645, 0.765055323929465, 1}},
        {6, {-1, -0.830223896278567, -0.468848793470714, 0.0, 0.468848793470714, 0.830223896278567, 1}},
        {7, {-1, -0.871740148509607, -0.591700181433142, -0.209299217902479, 0.2092992179024789, 0.591700181433142, 0.871740148509607, 1}},
        {8, {-1, -0.899757995411460, -0.677186279510738, -0.363117463826178, 0, 0.363117463826178, 0.677186279510738, 0.899757995411460, 1}}
    };
    
    std::vector<real> x;
    if (lgl.contains(n))
        x = lgl.at(n);
    else
    {
        // use the Golub-Welsch algorithm
        real* D = new real[n-1]{0};
        real* E = new real[n-2];
        for (u_long i=0; i < n-2; ++i)
        {
            real ii = i+1;
            E[i] = std::sqrt( ii * (ii + 2) / ((2*ii + 3) * (2*ii + 1)) );
        }

        int N = n-1;
        int info;
        char only_eigvals = 'N';
        int LDZ_dummy = 1;

        if constexpr(std::is_same<real,float>::value)
            ssteqr_(&only_eigvals, &N, D, E, nullptr, &LDZ_dummy, nullptr, &info);
        else
            dsteqr_(&only_eigvals, &N, D, E, nullptr, &LDZ_dummy, nullptr, &info);
        
        for (int i=0; i < N/2; ++i)
            x.push_back(D[i]);
        
        delete[] D;
        delete[] E;

        if (info != 0)
            throw std::runtime_error("gauss_lobatto() error: eigenvalue decomposition failed!");

        // refine roots with Newton's method
        for (real& z : x)
        {
            for (int i=0; i < 2; ++i) // just two iterations, shouldn't need much refinement
            {
                real q = boost::math::legendre_p<real>(n+1, z) - boost::math::legendre_p<real>(n-1, z);
                real dq = boost::math::legendre_p_prime<real>(n+1,z) - boost::math::legendre_p_prime<real>(n-1, z);
                z -= q/dq;
            }
        }

        if (n%2 == 0) // zero is one of the roots for (n+1) odd.
            x.push_back(0);

        // symmetry
        for (int i=N/2-1; i >= 0; --i)
            x.push_back(-x[i]);

        x.emplace(x.begin(), -1);
        x.push_back(1);
    }

    std::vector<real> w(n+1, 0);
    w[0] = 2.0 / (n*(n+1));
    w[n] = w[0];
    for (int i=0; i < n/2+1; ++i)
    {
        w[i] = 2.0 / (n*(n+1) * std::pow(boost::math::legendre_p(n, x[i]),2));
        w[n-i] = w[i];
    }

    return std::make_pair(std::move(x), std::move(w));
}

#endif