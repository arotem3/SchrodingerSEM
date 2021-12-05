#include <iostream>

bool test_gauss_lobatto();
bool test_derivative_matrix();
bool test_Quad();
bool test_glaplace();
bool test_load_mesh();
bool test_solution_wrapper();
bool test_pcg();
bool test_minres();
bool test_poisson();
bool test_helmholtz();

int main()
{
    bool success = true;
    
    success = test_gauss_lobatto()      && success;
    success = test_derivative_matrix()  && success;
    success = test_Quad()               && success;
    success = test_glaplace()           && success;
    success = test_load_mesh()          && success;
    success = test_solution_wrapper()   && success;
    success = test_pcg()                && success;
    success = test_minres()             && success;
    success = test_poisson()            && success;
    success = test_helmholtz()          && success;

    if (success)
        std::cout << "all tests passed! :)\n";

    return 0;
}