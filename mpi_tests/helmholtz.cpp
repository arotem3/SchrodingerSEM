#include "mpi_impl/helmholtz.hpp"
#include "mpi_impl/scatter_mesh.hpp"

namespace mpi = boost::mpi;
using namespace schro_mpi;

bool test_helmholtz(mpi::communicator& comm)
{
    auto f = [](double x, double y) -> double {return x*x - y*y;};

    int order = 6;
    auto [E2P, mesh] = scatter_mesh<double>(order, "../meshes/small_mesh", comm, 0);
    mesh.compute_metrics();

    int dof = mesh.dof();
    dof = mpi::all_reduce(comm, dof, std::plus<int>{});

    const auto& z = mesh.quadrature.x;

    SparseData<matrix<double>> F, u, c;
    for (auto& [e, element] : mesh.elements)
    {
        F[e] = arma::zeros(mesh.N, mesh.N);
        u[e] = arma::zeros(mesh.N, mesh.N);
        c[e] = arma::zeros(mesh.N, mesh.N);

        for (int i=0; i < mesh.N; ++i)
            for (int j=0; j < mesh.N; ++j)
            {
                auto [x, y] = mesh.elements[e].from_local_coo(z[i], z[j]);
                F[e](i, j) = f(x, y);
                c[e](i, j) = -100.0 - std::sqrt(2.0);
            }
    }

    auto rslts = helmholtz<double>(u, c, F, mesh, comm, E2P, dof*10, 1e-12);

    if (comm.rank() == 0)
        std::cout << "helmholtz returned after " << rslts.n_iter << " iteration with residual " << rslts.residual << std::endl;

    return rslts.success;
}