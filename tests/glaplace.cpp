#include "derivative_matrix.hpp"
#include "glaplace.hpp"

bool test_glaplace()
{
    quad_rule rule = gauss_lobatto<double>(5);

    Quad<double> element;
    element.xs[0] = -1; element.xs[1] = 1; element.xs[2] = 1; element.xs[3] = -1;
    element.ys[0] = -1; element.ys[1] = -1; element.ys[2] = 1; element.ys[3] = 1;

    element.compute_metrics(rule.x);

    arma::mat D = derivative_matrix(rule.x);

    int n = rule.x.size();
    arma::mat u = arma::ones(n, n);
    arma::mat lap_u = glaplace(u, element, D, rule);

    bool success = true;
    for (auto v : lap_u)
        success = success && (std::abs(v) < 1e-10);

    return success;
}