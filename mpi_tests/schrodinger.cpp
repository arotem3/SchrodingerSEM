#include "mpi_impl/schrodinger.hpp"
#include "mpi_impl/scatter_mesh.hpp"

using namespace schro_mpi;

double potential(double x, double y)
{
    return 0;
}

double u0(double x, double y)
{
    return std::sin(M_PI*x)*std::sin(M_PI*y);
}

bool test_schrodinger_bdf2(mpi::communicator& comm)
{
    int order = 3;
    Mesh<double> mesh = scatter_mesh<double>(order, "../meshes/small_mesh", comm, 0);
    mesh.compute_metrics();
    const auto& z = mesh.quadrature.x;

    int dof = mesh.dof();
    dof = mpi::all_reduce(comm, dof, std::plus<int>{});

    SparseData<matrix<double>> V, psi_real, psi_imag;
    for (auto& [e, element] : mesh.elements)
    {
        V[e] = arma::zeros(mesh.N, mesh.N);
        psi_real[e] = arma::zeros(mesh.N, mesh.N);
        psi_imag[e] = arma::zeros(mesh.N, mesh.N);
        for (int i=0; i < mesh.N; ++i)
            for (int j=0; j < mesh.N; ++j)
            {
                auto [x, y] = element.from_local_coo(z[i], z[j]);
                V[e](i, j) = potential(x, y);
                psi_real[e](i, j) = u0(x, y);
                psi_imag[e](i, j) = 0.0;
            }
    }

    double T = 0.01;

    solver_opts<double> opts;
    opts.dt = T / 10;
    opts.max_iter = 20*dof;
    opts.tol = 1e-8;
    opts.verbose = true;

    if (comm.rank() == 0)
        std::cout << "solving Schrodinger equation with BDF2...\n";
    bool success = schrodinger_bdf2<double>(psi_real, psi_imag, V, mesh, T, opts);

    if ((comm.rank() == 0) and (not success))
        std::cout << "schrodinger_bdf2 failed test\n";

    return success;
}

bool test_schrodinger_dirk4s3(mpi::communicator& comm)
{
    int order = 3;
    Mesh<double> mesh = scatter_mesh<double>(order, "../meshes/small_mesh", comm, 0);
    mesh.compute_metrics();
    const auto& z = mesh.quadrature.x;

    int dof = mesh.dof();
    dof = mpi::all_reduce(comm, dof, std::plus<int>{});

    SparseData<matrix<double>> V, psi_real, psi_imag;
    for (auto& [e, element] : mesh.elements)
    {
        V[e] = arma::zeros(mesh.N, mesh.N);
        psi_real[e] = arma::zeros(mesh.N, mesh.N);
        psi_imag[e] = arma::zeros(mesh.N, mesh.N);
        for (int i=0; i < mesh.N; ++i)
            for (int j=0; j < mesh.N; ++j)
            {
                auto [x, y] = element.from_local_coo(z[i], z[j]);
                V[e](i, j) = potential(x, y);
                psi_real[e](i, j) = u0(x, y);
                psi_imag[e](i, j) = 0.0;
            }
    }

    double T = 0.01;

    solver_opts<double> opts;
    opts.dt = T / 10;
    opts.max_iter = 20*dof;
    opts.tol = 1e-8;
    opts.verbose = true;

    if (comm.rank() == 0)
        std::cout << "solving Schrodinger equation with DIRK4S3...\n";
    bool success = schrodinger_dirk4s3<double>(psi_real, psi_imag, V, mesh, T, opts);

    if ((comm.rank() == 0) and (not success))
        std::cout << "schrodinger_dirk4s3 failed test\n";

    return success;
}

bool test_schrodinger_dirk4s5(mpi::communicator& comm)
{
    int order = 3;
    Mesh<double> mesh = scatter_mesh<double>(order, "../meshes/small_mesh", comm, 0);
    mesh.compute_metrics();
    const auto& z = mesh.quadrature.x;

    int dof = mesh.dof();
    dof = mpi::all_reduce(comm, dof, std::plus<int>{});

    SparseData<matrix<double>> V, psi_real, psi_imag;
    for (auto& [e, element] : mesh.elements)
    {
        V[e] = arma::zeros(mesh.N, mesh.N);
        psi_real[e] = arma::zeros(mesh.N, mesh.N);
        psi_imag[e] = arma::zeros(mesh.N, mesh.N);
        for (int i=0; i < mesh.N; ++i)
            for (int j=0; j < mesh.N; ++j)
            {
                auto [x, y] = element.from_local_coo(z[i], z[j]);
                V[e](i, j) = potential(x, y);
                psi_real[e](i, j) = u0(x, y);
                psi_imag[e](i, j) = 0.0;
            }
    }

    double T = 0.01;

    solver_opts<double> opts;
    opts.dt = T / 10;
    opts.max_iter = 20*dof;
    opts.tol = 1e-8;
    opts.verbose = true;

    if (comm.rank() == 0)
        std::cout << "solving Schrodinger equation with DIRK4S5...\n";
    bool success = schrodinger_dirk4s5<double>(psi_real, psi_imag, V, mesh, T, opts);

    if ((comm.rank() == 0) and (not success))
        std::cout << "schrodinger_dirk4s5 failed test\n";

    return success;
}