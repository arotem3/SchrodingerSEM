#include "Quad.hpp"

bool test_Quad()
{
    std::vector<double> x = {-1, 0, 1};

    Quad<double> element;
    element.xs[0] = -1; element.xs[1] = 1; element.xs[2] = 1; element.xs[3] = -1;
    element.ys[0] = -1; element.ys[1] = -1; element.ys[2] = 1; element.ys[3] = 1;

    element.compute_metrics(x);

    return true;
}