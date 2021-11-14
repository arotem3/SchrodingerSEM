#ifndef OP_HPP
#define OP_HPP

#include <concepts>

template <typename T>
concept Scalar = std::integral<T> || std::floating_point<T>;

template <typename ValueType>
class Plus
{
public:
    inline static ValueType apply(const ValueType& x, const ValueType& y)
    {
        return x + y;
    }
};

template <typename ValueType>
class Minus
{
public:
    inline static ValueType apply(const ValueType& x, const ValueType& y)
    {
        return x - y;
    }
};

template <Scalar scalar, typename ValueType>
class Scale
{
public:
    inline static ValueType apply(const scalar& s, const ValueType& x)
    {
        return s * x;
    }
};

#endif