#include "openmp_impl/helmholtz.hpp"
#include "load_mesh.hpp"

using namespace schro_omp;

bool test_helmholtz()
{
    auto f = [](double x, double y) -> double {return x*x - y*y;};

    int order = 6;
    Mesh<double> mesh = load_mesh<Mesh<double>>(order, "../meshes/small_mesh");
    mesh.compute_metrics();
    const auto& z = mesh.quadrature.x;

    int nel = mesh.elements.size();
    std::vector<matrix<double>> F(nel, arma::zeros(mesh.N, mesh.N)), c(nel, arma::zeros(mesh.N, mesh.N));
    for (int e=0; e < nel; ++e)
        for (int i=0; i < mesh.N; ++i)
            for (int j=0; j < mesh.N; ++j)
            {
                auto [x, y] = mesh.elements[e].from_local_coo(z[i], z[j]);
                F[e](i, j) = f(x, y);
                c[e](i, j) = -100.0 - std::sqrt(2.0);
            }

    std::vector<matrix<double>> u(nel, arma::zeros(mesh.N, mesh.N));
    auto rslts = helmholtz(u, c, F, mesh, mesh.dof()*10, 1e-10);

    std::cout << "helmholtz returned after " << rslts.n_iter << " iteration with residual " << rslts.residual << std::endl;

    return rslts.success;
}