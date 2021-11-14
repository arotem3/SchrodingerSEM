#ifndef TYPES_HPP
#define TYPES_HPP

// In this file we alias our data structures, so that they may be changed if
// neccessary.

#include <armadillo>

template <typename T>
using matrix = arma::Mat<T>;

template <typename T>
using mvec = arma::Col<T>;

#include <unordered_map>

template <typename T>
using SparseData = std::unordered_map<int, T>;

// maybe replace unordered_map with map?

#endif