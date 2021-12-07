#include "openmp_impl/schrodinger.hpp"
#include "load_mesh.hpp"

#include <omp.h>

using namespace schro_omp;

double potential(double x, double y)
{
    return 0;
}

double u0(double x, double y)
{
    return std::sin(M_PI*x)*std::sin(M_PI*y);
}

bool test_schrodinger()
{
    int order = 5;
    Mesh<double> mesh = load_mesh<Mesh<double>>(order, "../meshes/small_mesh");
    mesh.compute_metrics();
    const auto& z = mesh.quadrature.x;

    int nel = mesh.elements.size();
    std::vector<matrix<double>> V(nel, arma::zeros(mesh.N, mesh.N)), psi_real(nel, arma::zeros(mesh.N, mesh.N)), psi_nimag(nel, arma::zeros(mesh.N, mesh.N));
    for (int e=0; e < nel; ++e)
        for (int i=0; i < mesh.N; ++i)
            for (int j=0; j < mesh.N; ++j)
            {
                auto [x, y] = mesh.elements[e].from_local_coo(z[i], z[j]);
                V[e](i, j) = potential(x, y);
                psi_real[e](i, j) = u0(x, y);
                psi_nimag[e](i, j) = 0.0;
            }

    solver_opts<double> opts;
    opts.dt = 1.0 / M_PI / 400;
    opts.max_iter = 20*mesh.dof();
    opts.tol = 1e-12;

    int k = 0;
    auto callback = [&k, &mesh, &nel, &z](std::vector<matrix<double>> R, std::vector<matrix<double>> I) -> bool
    {
        // unmask(R, mesh); unmask(I, mesh);

        // std::ofstream out("sol/u" + std::to_string(k) + ".txt");
        // for (int e=0; e < nel; ++e)
        //     for (int i=0; i < mesh.N; ++i)
        //         for (int j=0; j < mesh.N; ++j)
        //         {
        //             auto [x, y] = mesh.elements[e].from_local_coo(z[i], z[j]);
        //             out << x << " " << y << " " << R[e](i,j) << " " << I[e](i,j) << std::endl;
        //         }
        // ++k;
        return false;
    };

    callback(psi_real, psi_nimag);

    double T = opts.dt * 200;
    std::cout << "running schrodinger solver...\n";
    bool success = schrodinger_bdf2<double>(psi_real, psi_nimag, V, mesh, T, opts, callback);
    // bool success = schrodinger_dirk4s3<double>(psi_real, psi_nimag, V, mesh, T, opts, callback);
    // bool success = schrodinger_dirk4s5<double>(psi_real, psi_nimag, V, mesh, T, opts, callback);

    if (not success)
        std::cout << "schrodinger failed test\n";

    return success;
}