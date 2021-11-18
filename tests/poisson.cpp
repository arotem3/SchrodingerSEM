#include "openmp_impl/poisson.hpp"
#include "load_mesh.hpp"

using namespace schro_omp;

double f(double x, double y)
{
    return x*x - y*y;
}

bool test_poisson()
{
    int order = 6;
    Mesh<double> mesh = load_mesh<Mesh<double>>(order, "../meshes/small_mesh");
    mesh.compute_metrics();
    const auto& z = mesh.quadrature.x;

    int nel = mesh.elements.size();
    std::vector<matrix<double>> F(nel, arma::zeros(mesh.N, mesh.N));
    for (int e=0; e < nel; ++e)
        for (int i=0; i < mesh.N; ++i)
            for (int j=0; j < mesh.N; ++j)
            {
                auto [x, y] = mesh.elements[e].from_local_coo(z[i], z[j]);
                F[e](i, j) = f(x, y);
            }

    std::vector<matrix<double>> u(nel, arma::zeros(mesh.N, mesh.N));
    auto rslts = poisson<double>(u, F, mesh, mesh.dof(), 1e-5);

    // std::cout << "poisson returned after " << rslts.n_iter << " iteration with residual " << rslts.residual << std::endl;

    return rslts.success;
}