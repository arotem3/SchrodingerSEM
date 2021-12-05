#include "mpi_impl/solution_wrapper.hpp"

using namespace schro_mpi;

bool test_solution_wrapper(mpi::communicator& comm)
{
    bool success = true;
    if (comm.rank() == 0) {
        std::unordered_map<int, arma::mat> m;
        m[1] = arma::ones(2,2);
        m[2] = arma::ones(2,2);
        m[4] = arma::ones(2,2);
        solution_wrapper<double> x(m);
        solution_wrapper<double> y = 0.5*x;

        x += 2.0 * (3.0*x - y);

        if (x.values[1][0] != 6.0) {
            std::cout << "solution_wrapper failed template expression computations\n";
            success = false;
        }

        bool size_preserved = m.size() == x.values.size();
        bool keys_preserved = std::equal(x.values.begin(), x.values.end(), m.begin(), [](auto a, auto b) { return a.first == b.first; });

        if (not (size_preserved and keys_preserved)) {
            std::cout << "solution_wrapper failed to preserve structure of original map\n";
            success = false;
        }
    }

    mpi::broadcast(comm, success, 0);

    return success;
}