#include <iostream>

bool test_gauss_lobatto();
bool test_derivative_matrix();

int main()
{
    bool success = true;
    
    success = success && test_gauss_lobatto();
    success = success && test_derivative_matrix();

    if (success)
        std::cout << "all tests passed! :)\n";

    return 0;
}