#include "mpi_impl/distribute_mesh.hpp"
#include "load_mesh.hpp"

using namespace schro_mpi;

bool test_distribute_mesh()
{
    int order = 3;
    Mesh<double> mesh = load_mesh<Mesh<double>>(order, "../meshes/small_mesh");
    int P = 4;
    std::unordered_map<int, int> elem2proc = distribute_mesh(mesh, P);

    return true;
}