#ifndef MPI_IMPL_V_WRAPPER_HPP
#define MPI_IMPL_V_WRAPPER_HPP

#include <vector>

#include "mpi_impl/solution_wrapper.hpp"

namespace schro_mpi
{
    // (TEMPORARY SOLUTION) TO DO: FIX TEMPLATE EXPRESSIONS
    template <std::floating_point real>
    class v_wrapper
    {
    public:
        std::vector<solution_wrapper<real>> values;

        inline size_t size() const
        {
            return values.size();
        }

        inline const solution_wrapper<real>& operator[](size_t i) const
        {
            return values[i];
        }

        inline solution_wrapper<real>& operator[](size_t i)
        {
            return values[i];
        }

        v_wrapper() {}

        v_wrapper(const v_wrapper<real>& a) : values(a.values) {}
        v_wrapper<real>& operator=(const v_wrapper<real>& a)
        {
            values = a.values;
            return *this;
        }

        v_wrapper(v_wrapper<real>&& a) : values(std::move(a.values)) {}
        v_wrapper<real>& operator=(v_wrapper<real>&& a)
        {
            values = std::move(a.values);
            return *this;
        }

        v_wrapper<real>& operator+=(const v_wrapper<real>& a)
        {
            for (int i=0; i < a.size(); ++i)
                values.at(i) += a[i];

            return *this;
        }

        v_wrapper<real>& operator-=(const v_wrapper<real>& a)
        {
            for (int i=0; i < a.size(); ++i)
                values.at(i) -= a[i];

            return *this;
        }
    };

    template <std::floating_point real>
    v_wrapper<real> operator+(const v_wrapper<real>& a, const v_wrapper<real>& b)
    {
        v_wrapper<real> c = a;
        c += b;

        return c;
    }

    template <std::floating_point real>
    v_wrapper<real> operator-(const v_wrapper<real>& a, const v_wrapper<real>& b)
    {
        v_wrapper<real> c = a;
        c -= b;

        return c;
    }

    template <std::floating_point real>
    v_wrapper<real> operator*(real s, const v_wrapper<real>& x)
    {
        v_wrapper<real> y;
        for (int i=0; i < x.size(); ++i)
            y.values.push_back(solution_wrapper<real>(s * x[i]));
        
        return y;
    }

    template <std::floating_point real>
    v_wrapper<real> operator*(const v_wrapper<real>& x, real s)
    {
        v_wrapper<real> y;
        for (int i=0; i < x.size(); ++i)
            y.values.push_back(solution_wrapper<real>(s * x[i]));
        
        return y;
    }
} // namespace schro_mpi


// namespace schro_mpi
// {
//     template <typename T, typename ValueType>
//     class vExpr
//     {
//     public:
//         inline size_t size() const
//         {
//             return static_cast<const T&>(*this).size();
//         }

//         inline ValueType operator[](size_t i) const
//         {
//             return static_cast<const T&>(*this)[i];
//         }
//     };

//     template <typename T1, typename T2, typename Op, typename ValueType>
//     class vBinaryExpr : public vExpr<vBinaryExpr<T1, T2, Op, ValueType>, ValueType>
//     {
//     private:
//         const T1& _x;
//         const T2& _y;

//     public:
//         vBinaryExpr(const T1& x, const T2& y) : _x(x), _y(y) {}

//         inline ValueType operator[](size_t i) const
//         {
//             return Op::apply(_x[i], _y[i]);
//         }

//         inline size_t size() const
//         {
//             return _x.size();
//         }
//     };

//     template <Scalar scalar, typename T, typename Op, typename ValueType>
//     class vScalingExpr : public vExpr<vScalingExpr<scalar, T, Op, ValueType>, ValueType>
//     {
//     private:
//         scalar _s;
//         const T& _x;

//     public:
//         vScalingExpr(scalar s, const T& x) : _s(s), _x(x) {}

//         inline ValueType operator[](size_t i) const
//         {
//             return Op::apply(_s, _x[i]);
//         }

//         inline size_t size() const
//         {
//             return _x.size();
//         }
//     };

//     template <typename T1, typename T2, typename ValueType>
//     vBinaryExpr<T1, T2, Plus<ValueType>, ValueType>
//     operator+(const vExpr<T1, ValueType>& x, const vExpr<T2, ValueType>& y)
//     {
//         return vBinaryExpr<T1, T2, Plus<ValueType>, ValueType>(*static_cast<const T1*>(&x), *static_cast<const T2*>(&y));
//     }

//     template <typename T1, typename T2, typename ValueType>
//     vBinaryExpr<T1, T2, Minus<ValueType>, ValueType>
//     operator-(const vExpr<T1, ValueType>& x, const vExpr<T2, ValueType>& y)
//     {
//         return vBinaryExpr<T1, T2, Minus<ValueType>, ValueType>(*static_cast<const T1*>(&x), *static_cast<const T2*>(&y));
//     }

//     template <Scalar scalar, typename T, typename ValueType>
//     vScalingExpr<scalar, T, Scale<scalar, ValueType>, ValueType>
//     operator*(scalar s, const vExpr<T, ValueType>& x)
//     {
//         return vScalingExpr<scalar, T, Scale<scalar, ValueType>, ValueType>(s, *static_cast<const T*>(&x));
//     }

//     template <Scalar scalar, typename T, typename ValueType>
//     vScalingExpr<scalar, T, Scale<scalar, ValueType>, ValueType>
//     operator*(const vExpr<T, ValueType>& x, scalar s)
//     {
//         return vScalingExpr<scalar, T, Scale<scalar, ValueType>, ValueType>(s, *static_cast<const T*>(&x));
//     }

//     template <std::floating_point real>
//     class v_wrapper : public vExpr<v_wrapper<real>, solution_wrapper<real>>
//     {
//     public:
//         std::vector<solution_wrapper<real>> values;

//         inline size_t size() const
//         {
//             return values.size();
//         }

//         inline const solution_wrapper<real>& operator[](size_t i) const
//         {
//             return values[i];
//         }

//         inline solution_wrapper<real>& operator[](size_t i)
//         {
//             return values[i];
//         }
    
//         v_wrapper() {}

//         v_wrapper(const v_wrapper<real>& a) : values(a.values) {}
//         v_wrapper<real>& operator=(const v_wrapper<real>& a)
//         {
//             values = a.values;
//             return *this;
//         }

//         v_wrapper(v_wrapper<real>&& a) : values(std::move(a.values)) {}
//         v_wrapper<real>& operator=(v_wrapper<real>&& a)
//         {
//             values = std::move(a.values);
//             return *this;
//         }

//         template <typename T>
//         v_wrapper(const vExpr<T, solution_wrapper<real>>& a) : values(a.size())
//         {
//             for (int i=0; i < a.size(); ++i)
//                 values[i] = a[i];
//         }
//         template <typename T>
//         v_wrapper<real>& operator=(const vExpr<T, solution_wrapper<real>>& a)
//         {
//             if (values.size() != a.size())
//                 values.resize(a.size());
            
//             for (int i=0; i < a.size(); ++i)
//                 values[i] = a[i];

            
//             return *this;
//         }

//         template <typename T>
//         v_wrapper<real>& operator+=(const vExpr<T, solution_wrapper<real>>& a)
//         {
//             for (int i=0; i < a.size(); ++i)
//                 values[i] += a[i];

//             return *this;
//         }

//         template <typename T>
//         v_wrapper<real>& operator-=(const vExpr<T, solution_wrapper<real>>& a)
//         {
//             for (int i=0; i < a.size(); ++i)
//                 values[i] -= a[i];
            
//             return *this;
//         }
//     };
// } // namespace schro_mpi


#endif