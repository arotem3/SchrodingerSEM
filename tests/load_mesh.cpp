#include "load_mesh.hpp"

bool test_load_mesh()
{
    int order = 3;
    Mesh<double> mesh1 = load_mesh<Mesh<double>>(order, "../meshes/small_mesh");
    SpMesh<double> mesh2 =load_mesh<SpMesh<double>>(order, "../meshes/small_mesh");

    return true;
}