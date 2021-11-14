#include "solution_wrapper.hpp"

bool test_solution_wrapper()
{
    int n = 3;
    solution_wrapper<double> x(std::vector<arma::mat>(n, arma::ones(2,2)));
    solution_wrapper<double> y = 0.5*x;

    x += 2.0 * (3.0*x - y);
    
    bool success = true;

    if (x[0][0] != 6.0) {
        std::cout << "solution_wrapper failed template expression computations\n";
        success = false;
    }

    if (x.size() != n) {
        std::cout << "solution_wrapper failed to preserve structure of original vector\n";
        success = false;
    }

    return success;
}