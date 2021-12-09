#ifndef OMP_IMPL_SOLUTION_WRAPPER_HPP
#define OMP_IMPL_SOLUTION_WRAPPER_HPP

#include <vector>

#include "types.hpp"
#include "Op.hpp"

namespace schro_omp
{
    // implements template expressions for vector-vector operations {+,-} and
    // vector-scalar operation {*} to minimize extra memory usage and to clean up
    // code down the line.
    // see:
    // https://en.wikipedia.org/wiki/Expression_templates
    // https://web.archive.org/web/20050210090012/http://osl.iu.edu/~tveldhui/papers/Expression-Templates/exprtmpl.html

    // class implements abstract expressions for vector-valued operations. T is the
    // operation, and ValueType is the type returned on index.
    template <typename T, typename ValueType>
    class Expr
    {
    public:
        inline size_t size() const
        {
            return static_cast<const T&>(*this).size();
        }

        inline ValueType operator[](size_t i) const
        {
            return static_cast<const T&>(*this)[i];
        }
    };

    template <typename T1, typename T2, typename Op, typename ValueType>
    class BinaryExpr : public Expr<BinaryExpr<T1, T2, Op, ValueType>, ValueType>
    {
    private:
        const T1& _x;
        const T2& _y;

    public:
        BinaryExpr(const T1& x, const T2& y) : _x(x), _y(y) {}

        inline ValueType operator[](size_t i) const
        {
            return Op::apply(_x[i], _y[i]);
        }

        inline size_t size() const
        {
            return _x.size();
        }
    };

    template <Scalar scalar, typename T, typename Op, typename ValueType>
    class ScalingExpr : public Expr<ScalingExpr<scalar, T, Op, ValueType>, ValueType>
    {
    private:
        scalar _s;
        const T& _x;

    public:
        ScalingExpr(scalar s, const T& x) : _s(s), _x(x) {}

        inline ValueType operator[](size_t i) const
        {
            return Op::apply(_s, _x[i]);
        }

        inline size_t size() const
        {
            return _x.size();
        }
    };

    template <typename T1, typename T2, typename ValueType>
    BinaryExpr<T1, T2, Plus<ValueType>, ValueType>
    operator+(const Expr<T1, ValueType>& x, const Expr<T2, ValueType>& y)
    {
        return BinaryExpr<T1, T2, Plus<ValueType>, ValueType>(*static_cast<const T1*>(&x), *static_cast<const T2*>(&y));
    }

    template <typename T1, typename T2, typename ValueType>
    BinaryExpr<T1, T2, Minus<ValueType>, ValueType>
    operator-(const Expr<T1, ValueType>& x, const Expr<T2, ValueType>& y)
    {
        return BinaryExpr<T1, T2, Minus<ValueType>, ValueType>(*static_cast<const T1*>(&x), *static_cast<const T2*>(&y));
    }

    template <Scalar scalar, typename T, typename ValueType>
    ScalingExpr<scalar, T, Scale<scalar, ValueType>, ValueType>
    operator*(scalar s, const Expr<T, ValueType>& x)
    {
        return ScalingExpr<scalar, T, Scale<scalar, ValueType>, ValueType>(s, *static_cast<const T*>(&x));
    }

    template <Scalar scalar, typename T, typename ValueType>
    ScalingExpr<scalar, T, Scale<scalar, ValueType>, ValueType>
    operator*(const Expr<T, ValueType>& x, scalar s)
    {
        return ScalingExpr<scalar, T, Scale<scalar, ValueType>, ValueType>(s, *static_cast<const T*>(&x));
    }

    // wrapper for PDE solutions that enables vector-vector operations {+,+=,-,-=}
    // and vector-scalar operation {*} in compatibility with linear solvers
    template <std::floating_point real>
    class solution_wrapper : public Expr<solution_wrapper<real>, matrix<real>>
    {
    public:
        typedef real value_type;

        std::vector<matrix<real>> values;

        solution_wrapper() {}

        // move vector
        solution_wrapper(std::vector<matrix<real>>&& a) : values(std::move(a)) {}
        solution_wrapper<real>& operator=(std::vector<matrix<real>>&& a)
        {
            values = std::move(a);
            return *this;
        }

        // copy vector
        solution_wrapper(const std::vector<matrix<real>>& a) : values(a) {}
        solution_wrapper<real>& operator=(const std::vector<matrix<real>>& a)
        {
            values = a;
            return *this;
        }

        // move
        solution_wrapper(solution_wrapper<real>&& a) : values(std::move(a.values)) {}
        solution_wrapper<real>& operator=(solution_wrapper<real>&& a)
        {
            values = std::move(a.values);
            return *this;
        }

        // copy
        solution_wrapper(const solution_wrapper<real>& a) : values(a.values) {}
        solution_wrapper<real>& operator=(const solution_wrapper<real>& a)
        {
            values = a.values;
            return *this;
        }

        // evaluate expression
        template <typename T>
        solution_wrapper(const Expr<T, matrix<real>>& op) : values(op.size())
        {
            #pragma omp parallel for if(op.size() > 63) schedule(static) 
            for (int i=0; i < op.size(); ++i)
                values[i] = op[i];
        }
        template <typename T>
        solution_wrapper<real>& operator=(const Expr<T, matrix<real>>& op)
        {
            values.resize(op.size());
            #pragma omp parallel for if(op.size() > 63) schedule(static) 
            for (int i=0; i < op.size(); ++i)
                values[i] = op[i];
            return *this;
        }

        template <typename T>
        solution_wrapper<real>& operator+=(const Expr<T, matrix<real>>& op)
        {
            #pragma omp parallel for if(op.size() > 63) schedule(static) 
            for (int i=0; i < op.size(); ++i)
                values[i] += op[i];

            return *this;
        }

        template <typename T>
        solution_wrapper<real>& operator-=(const Expr<T, matrix<real>>& op)
        {
            #pragma omp parallel for if(op.size() > 63) schedule(static) 
            for (int i=0; i < op.size(); ++i)
                values[i] -= op[i];

            return *this;
        }

        inline const matrix<real>& operator[](size_t i) const
        {
            return values[i];
        }

        inline matrix<real>& operator[](size_t i)
        {
            return values[i];
        }

        inline size_t size() const
        {
            return values.size();
        }
    };
} // namespace schro_omp
#endif