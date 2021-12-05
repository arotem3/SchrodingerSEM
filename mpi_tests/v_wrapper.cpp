#include "mpi_impl/v_wrapper.hpp"

using namespace schro_mpi;

bool test_v_wrapper(mpi::communicator& comm)
{
    bool success = true;
    if (comm.rank() == 0) {
        std::unordered_map<int, arma::mat> u;
        std::unordered_map<int, arma::mat> v;
        for (int i : {1, 2, 4})
        {
            u[i] = arma::ones(2,2);
            v[i] = 2.0*arma::ones(2,2);
        }

        v_wrapper<double> U;
        U.values.push_back(u);
        U.values.push_back(std::move(v));

        v_wrapper<double> W = 0.5*U;

        U += 2.0 * (3.0*U - W);

        if ((U[0].values[1][0] != 6.0) or (U[1].values[1][0] != 12.0)) {
            std::cout << "v_wrapper failed template expression computations\n";
            success = false;
        }

        bool size_preserved = (U.size() == 2) and (U[0].values.size() == 3) and (U[1].values.size() == 3);
        bool keys_preserved = std::equal(U[0].values.begin(), U[0].values.end(), u.begin(), [](auto a, auto b){return a.first == b.first;});

        if (not (size_preserved and keys_preserved)) {
            std::cout << "v_wrapper failed to preserve structure of original maps\n";
            success = false;
        }
    }

    mpi::broadcast(comm, success, 0);

    return success;
}