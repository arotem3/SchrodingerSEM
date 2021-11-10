#include "derivative_matrix.hpp"
#include "gauss_lobatto.hpp"

arma::vec p(const arma::vec& x)
{
    return -1 + 2*x + 5*x%x - 3*x%x%x;
}

arma::vec dp(const arma::vec& x)
{
    return 2 + 10*x - 9*x%x;
}

bool test_derivative_matrix()
{
    auto [x, w] = gauss_lobatto<double>(8);

    arma::mat D = derivative_matrix<double>(x);

    arma::vec t = arma::conv_to<arma::vec>::from(x);
    arma::vec y = p(t);
    arma::vec dy = dp(t);

    double e = arma::norm(D*y - dy, "inf"); // y is a polynomial of degree <= 8 so should be exact (upto rounding errors)

    bool passed = true;
    
    if (e > 1e-10)
    {
        passed = false;
        std::cout << "derivative_matrix() failed accuracy test\n";
    }
    
    return passed;
}