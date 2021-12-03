#include "load_mesh.hpp"
#include "openmp_impl/Mesh.hpp"
// #include "mpi_impl/Mesh.hpp"

bool test_load_mesh()
{
    int order = 3;
    // schro_mpi::Mesh<double> mesh1 = load_mesh<schro_mpi::Mesh<double>>(order, "../meshes/small_mesh");
    schro_omp::Mesh<double> mesh2 =load_mesh<schro_omp::Mesh<double>>(order, "../meshes/small_mesh");

    return true;
}