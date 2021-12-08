#ifndef VERBOSE_HPP
#define VERBOSE_HPP

#include <concepts>
#include <string>
#include <iostream>
#include <iomanip>

template <std::floating_point real>
std::string progress_bar(real t, real T)
{
    const int n = 30;
    int p = n * (t/T);

    std::string out(n+2, ' ');
    out[0] = '[';
    for (int i=0; i < p; ++i)
        out[i+1] = '-';
    out[p+1] = '>';
    out[n+1] = ']';

    return out;
}

class VerboseTracker
{
public:
    int precision;
    std::ostream& out;

    VerboseTracker(std::ostream& out = std::cout, int precision = 2) : out(out), precision(precision) {}

    template <std::floating_point real>
    void operator()(real t, real T)
    {
        out << std::fixed << std::setprecision(precision) << t << " :: " << progress_bar(t, T) << '\r' << std::flush;
    }

    template <std::floating_point real>
    void operator()(real t, real T, real e)
    {
        out << std::fixed << std::setprecision(precision) << t << " :: " << progress_bar(t, T) << std::scientific << " :: error ~ " << e << '\r' << std::flush;
    }
};

#endif