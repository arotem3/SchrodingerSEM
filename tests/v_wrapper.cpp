#include "openmp_impl/v_wrapper.hpp"

using namespace schro_omp;

bool test_v_wrapper()
{
    int n = 3;
    std::vector<arma::mat> u(n, arma::ones(2,2));
    std::vector<arma::mat> v(n, 2*arma::ones(2,2));

    v_wrapper<double> U;
    U.values.push_back(std::move(u));
    U.values.push_back(std::move(v));

    v_wrapper<double> W = 0.5*U;

    U += 2.0 * (3.0*U - W);

    bool success = true;

    if ((U[0][0][0] != 6.0) or (U[1][0][0] != 12.0)) {
        std::cout << "v_wrapper failed template expression computations\n";
        success = false;
    }

    if ((U.size() != 2) or (U[0].size() != n) or (U[1].size() != n)) {
        std::cout << "v_wrapper failed to preserve structure of original vectors\n";
        success = false;
    }
    
    return success;
}