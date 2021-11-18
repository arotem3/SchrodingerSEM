#include <boost/mpi.hpp>
#include "mpi_impl/scatter_mesh.hpp"
#include "mpi_impl/poisson.hpp"
#include "load_mesh.hpp"

namespace mpi = boost::mpi;
using namespace schro_mpi;

double f(double x, double y)
{
    return x*x - y*y;
}

bool test_poisson(mpi::communicator& comm)
{
    int order = 6;
    
    auto [E2P, mesh] = scatter_mesh<double>(order, "../meshes/small_mesh", comm, 0);
    mesh.compute_metrics();

    const auto& z = mesh.quadrature.x;

    SparseData<matrix<double>> F, u;
    for (auto& [e, element] : mesh.elements)
    {
        F[e] = arma::zeros(mesh.N, mesh.N);
        u[e] = arma::zeros(mesh.N, mesh.N);

        for (int i=0; i < mesh.N; ++i)
            for (int j=0; j < mesh.N; ++j)
            {
                auto [x, y] = mesh.elements[e].from_local_coo(z[i], z[j]);
                F[e](i, j) = f(x, y);
            }
    }
    
    auto rslts = poisson<double>(u, F, mesh, comm, E2P, 2'000, 1e-5);

    if (comm.rank() == 0)
        std::cout << "poisson returned after " << rslts.n_iter << " iteration with residual " << rslts.residual << std::endl;

    return rslts.success;
}