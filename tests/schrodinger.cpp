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

bool test_schrodinger_bdf2()
{
    int order = 3;
    Mesh<double> mesh = load_mesh<Mesh<double>>(order, "../meshes/small_mesh");
    mesh.compute_metrics();
    const auto& z = mesh.quadrature.x;

    int nel = mesh.elements.size();
    std::vector<matrix<double>> V(nel, arma::zeros(mesh.N, mesh.N)), psi_real(nel, arma::zeros(mesh.N, mesh.N)), psi_imag(nel, arma::zeros(mesh.N, mesh.N));
    for (int e=0; e < nel; ++e)
        for (int i=0; i < mesh.N; ++i)
            for (int j=0; j < mesh.N; ++j)
            {
                auto [x, y] = mesh.elements[e].from_local_coo(z[i], z[j]);
                V[e](i, j) = potential(x, y);
                psi_real[e](i, j) = u0(x, y);
                psi_imag[e](i, j) = 0.0;
            }

    double T = 0.01;

    solver_opts<double> opts;
    opts.dt = T / 10;
    opts.max_iter = 20*mesh.dof();
    opts.tol = 1e-8;
    opts.verbose = true;

    std::cout << "solving Schrodinger equation with BDF2...\n";
    bool success = schrodinger_bdf2<double>(psi_real, psi_imag, V, mesh, T, opts);

    if (not success)
        std::cout << "schrodinger_df2 failed test\n";

    return success;
}

bool test_schrodinger_dirk4s3()
{
    int order = 3;
    Mesh<double> mesh = load_mesh<Mesh<double>>(order, "../meshes/small_mesh");
    mesh.compute_metrics();
    const auto& z = mesh.quadrature.x;

    int nel = mesh.elements.size();
    std::vector<matrix<double>> V(nel, arma::zeros(mesh.N, mesh.N)), psi_real(nel, arma::zeros(mesh.N, mesh.N)), psi_imag(nel, arma::zeros(mesh.N, mesh.N));
    for (int e=0; e < nel; ++e)
        for (int i=0; i < mesh.N; ++i)
            for (int j=0; j < mesh.N; ++j)
            {
                auto [x, y] = mesh.elements[e].from_local_coo(z[i], z[j]);
                V[e](i, j) = potential(x, y);
                psi_real[e](i, j) = u0(x, y);
                psi_imag[e](i, j) = 0.0;
            }

    double T = 0.01;

    solver_opts<double> opts;
    opts.dt = T/10;
    opts.max_iter = 20*mesh.dof();
    opts.tol = 1e-8;
    opts.verbose = true;

    std::cout << "solving schrodinger equation with dirk4s3...\n";
    bool success = schrodinger_dirk4s3<double>(psi_real, psi_imag, V, mesh, T, opts);

    if (not success)
        std::cout << "schrodinger_dirk4s3 failed test\n";

    return success;
}

bool test_schrodinger_dirk4s5()
{
    int order = 3;
    Mesh<double> mesh = load_mesh<Mesh<double>>(order, "../meshes/small_mesh");
    mesh.compute_metrics();
    const auto& z = mesh.quadrature.x;

    int nel = mesh.elements.size();
    std::vector<matrix<double>> V(nel, arma::zeros(mesh.N, mesh.N)), psi_real(nel, arma::zeros(mesh.N, mesh.N)), psi_imag(nel, arma::zeros(mesh.N, mesh.N));
    for (int e=0; e < nel; ++e)
        for (int i=0; i < mesh.N; ++i)
            for (int j=0; j < mesh.N; ++j)
            {
                auto [x, y] = mesh.elements[e].from_local_coo(z[i], z[j]);
                V[e](i, j) = potential(x, y);
                psi_real[e](i, j) = u0(x, y);
                psi_imag[e](i, j) = 0.0;
            }

    double T = 0.01;

    solver_opts<double> opts;
    opts.dt = T / 10;
    opts.max_iter = 20*mesh.dof();
    opts.tol = 1e-8;
    opts.verbose = true;
    
    std::cout << "solving schrodinger equation with dirk4s5...\n";
    bool success = schrodinger_dirk4s5<double>(psi_real, psi_imag, V, mesh, T, opts);

    if (not success)
        std::cout << "schrodinger_dirk4s5 failed test\n";

    return success;
}