#include "mpi_impl/scatter_mesh.hpp"

using namespace schro_mpi;

bool test_scatter_mesh(mpi::communicator& comm)
{
    int order = 3;
    Mesh<double> mesh = scatter_mesh<double>(order, "../meshes/small_mesh", comm, 0);

    return true;
}