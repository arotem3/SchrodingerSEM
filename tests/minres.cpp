#include <minres.hpp>

#include <armadillo>

bool test_minres()
{
    // finite difference helmholtz: -u''(x) - k*u(x) == x^2, u(-1) == u(1) == 0 {k is chosen so the system is indefinite}
    int n = 20;
    double k = 2;
    arma::vec x = arma::linspace(-1, 1, n);
    double h = x[1] - x[0];
    arma::vec D2 = {-1.0, 2.0, -1.0};
    D2 /= h*h;
    auto L = [&D2,k](const arma::vec& u) -> arma::vec
    {
        return arma::conv(u, D2, "same") - k*u;
    };
    arma::vec f = x%x;

    arma::vec u = arma::zeros(n);
    double tol = 1e-5;
    auto rslts = minres(u, L, f, arma::dot<arma::vec,arma::vec>, IdentityPreconditioner{}, n, tol);

    bool success = true;
    if (arma::norm(L(u) - f) > tol * arma::norm(f)) {
        std::cout << "pcg returned bad residual\n";
        success = false;
    }

    return success;
}