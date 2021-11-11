#include "distribute_mesh.hpp"
#include "load_mesh.hpp"

bool test_distribute_mesh()
{
    int order = 3;
    SpMesh<double> mesh = load_mesh<SpMesh<double>>(order, "../meshes/small_mesh");
    int P = 4;
    std::unordered_map<int, int> elem2proc = distribute_mesh(mesh, P);

    return true;
}