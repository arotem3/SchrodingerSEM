#include <iostream>
#include <boost/math/special_functions/bessel.hpp>

#include "gauss_lobatto.hpp"

double f(double x)
{
    return std::exp(std::sin(M_PI*x));
}

bool test_gauss_lobatto()
{
    auto [x, w] = gauss_lobatto<double>(9);

    double I = 0;
    for (int i=0; i < x.size(); ++i)
        I += f(x[i])*w[i];
    
    double e1 = std::abs(I - 2*boost::math::cyl_bessel_i(0,1.0));

    auto [y, c] = gauss_lobatto<double>(10);

    I = 0;
    for (int i=0; i < y.size(); ++i)
        I += f(y[i]) * c[i];
    
    double e2 = std::abs(I - 2*boost::math::cyl_bessel_i(0,1.0));

    bool passed = true;

    if (e1 > 1e-4)
    {
        std::cout << "gauss_lobatto() failed accuracy test.\n";
        passed = false;
    }

    if (e2 > e1)
    {
        std::cout << "gauss_lobatto() failed convergence test.\n";        
        passed = false;
    }

    return passed;
}