#include <iostream>
#include <boost/mpi.hpp>

namespace mpi = boost::mpi;

bool test_poisson(mpi::communicator&);
bool test_dot(mpi::communicator&);
bool test_solution_wrapper(mpi::communicator&);
bool test_scatter_mesh(mpi::communicator&);
bool test_helmholtz(mpi::communicator&);
bool test_v_wrapper(mpi::communicator&);

int main(int argc, char* argv[])
{
    mpi::environment env(argc, argv);
    mpi::communicator comm;

    bool success = true;

    success = test_dot(comm)                && success;
    success = test_solution_wrapper(comm)   && success;
    success = test_scatter_mesh(comm)       && success;
    success = test_poisson(comm)            && success;
    success = test_helmholtz(comm)          && success;
    success = test_v_wrapper(comm)          && success;

    if (success)
        if (comm.rank() == 0)
            std::cout << "all tests passed :)\n";

    return 0;
}