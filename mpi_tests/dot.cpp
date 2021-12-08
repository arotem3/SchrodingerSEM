#include <boost/mpi.hpp>

#include "mpi_impl/scatter_mesh.hpp"
#include "mpi_impl/dot.hpp"

namespace mpi = boost::mpi;
using namespace schro_mpi;

bool test_dot(mpi::communicator& comm)
{
    int order = 6;
    
    Mesh<double> mesh = scatter_mesh<double>(order, "../meshes/small_mesh", comm, 0);
    mesh.compute_metrics();

    SparseData<matrix<double>> u;
    for (auto it = mesh.elements.begin(); it != mesh.elements.end(); ++it)
        u[it->first] = arma::ones(mesh.N, mesh.N);

    double p = dot(mesh, u, u); // since u = 1, dot(u,u) = total degrees of freedom

    int d = mesh.dof(); // processor-local degrees of freedom
    d = mpi::all_reduce(comm, d, std::plus<int>{}); // all degrees of freedom

    bool success = true;
    if (p != d) {
        if (comm.rank() == 0)
            std::cout << "dot() failed test: produced incorrect value\n";
        success = false;
    }

    return success;
}