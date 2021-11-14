#include "sp_solution_wrapper.hpp"

bool test_sp_solution_wrapper()
{
    std::unordered_map<int, arma::mat> m;
    m[1] = arma::ones(2,2);
    m[2] = arma::ones(2,2);
    m[4] = arma::ones(2,2);
    sp_solution_wrapper<double> x(m);
    sp_solution_wrapper<double> y = 0.5*x;

    bool success = true;

    x += 2.0 * (3.0*x - y);

    if (x.values[1][0] != 6.0) {
        std::cout << "sp_solutio_wrapper failed template expression computations\n";
        success = false;
    }

    bool size_preserved = m.size() == x.values.size();
    bool keys_preserved = std::equal(x.values.begin(), x.values.end(), m.begin(), [](auto a, auto b) { return a.first == b.first; });

    if (not (size_preserved and keys_preserved)) {
        std::cout << "sp_solution_wrapper failed to preserve structure of original map\n";
        success = false;
    }

    return success;
}