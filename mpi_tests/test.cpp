#include <iostream>
#include <boost/mpi.hpp>

namespace mpi = boost::mpi;

bool test_poisson(mpi::communicator& comm);

int main(int argc, char* argv[])
{
    mpi::environment env(argc, argv);
    mpi::communicator comm;

    bool success = true;

    success = success && test_poisson(comm);

    if (success)
        std::cout << "all tests passed :)\n";

    return 0;
}