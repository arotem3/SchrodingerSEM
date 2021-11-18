#ifndef MPI_IMPL_SOLUTION_WRAPPER_HPP
#define MPI_IMPL_SOLUTION_WRAPPER_HPP

#include <unordered_map>
#include <utility>

#include "types.hpp"
#include "mpi_impl/Mesh.hpp"
#include "Op.hpp"

namespace schro_mpi
{
    // implements template expressions for vector-vector operations {+,-} and
    // vector-scalar operation {*} to minimize extra memory usage and to clean up
    // code down the line. This version is for the sparse data structures used in
    // the distributed version of the algorithms
    // see:
    // https://en.wikipedia.org/wiki/Expression_templates
    // https://web.archive.org/web/20050210090012/http://osl.iu.edu/~tveldhui/papers/Expression-Templates/exprtmpl.html 

    template <typename T, typename ValueType>
    class Expr
    {
    private:
        T _it;

    public:
        typedef std::pair<int, ValueType> pT;

        Expr(const T& it) : _it(it) {}

        inline pT operator*() const
        {
            return *_it;
        }

        inline void operator++()
        {
            ++_it;
        }

        inline size_t size() const
        {
            return _it.size();
        }
    };

    template <typename T1, typename T2, typename Op, typename ValueType>
    class BinaryExpr
    {
    private:
        T1 _x;
        T2 _y;

    public:
        typedef std::pair<int, ValueType> pT;

        BinaryExpr(const T1& x, const T2& y) : _x(x), _y(y) {}

        inline pT operator*() const
        {
            pT x = *_x;
            pT y = *_y;
            x.second = Op::apply(x.second, y.second);
            return x;
        }

        void operator++()
        {
            ++_x;
            ++_y;
        }

        inline size_t size() const
        {
            return _x.size();
        }
    };

    template <Scalar scalar, typename T, typename Op, typename ValueType>
    class ScalingExpr
    {
    private:
        scalar _s;
        T _x;

    public:
        typedef std::pair<int, ValueType> pT;
        
        ScalingExpr(scalar s, const T& x) : _s(s), _x(x) {}

        inline pT operator*() const
        {
            pT x = *_x;
            x.second = Op::apply(_s, x.second);
            return x;
        }

        void operator++()
        {
            ++_x;
        }

        inline size_t size() const
        {
            return _x.size();
        }
    };

    template <typename T1, typename T2, typename ValueType>
    Expr<BinaryExpr<Expr<T1, ValueType>, Expr<T2, ValueType>, Plus<ValueType>, ValueType>, ValueType>
    operator+(const Expr<T1, ValueType>& x, const Expr<T2, ValueType>& y)
    {
        typedef BinaryExpr<Expr<T1, ValueType>, Expr<T2, ValueType>, Plus<ValueType>, ValueType> bexp;
        return Expr<bexp, ValueType>(bexp(x, y));
    }

    template <typename T1, typename T2, typename ValueType>
    Expr<BinaryExpr<Expr<T1, ValueType>, Expr<T2, ValueType>, Minus<ValueType>, ValueType>, ValueType>
    operator-(const Expr<T1, ValueType>& x, const Expr<T2, ValueType>& y)
    {
        typedef BinaryExpr<Expr<T1, ValueType>, Expr<T2, ValueType>, Minus<ValueType>, ValueType> bexp;
        return Expr<bexp, ValueType>(bexp(x, y));
    }

    template <Scalar scalar, typename T, typename ValueType>
    Expr<ScalingExpr<scalar, Expr<T, ValueType>, Scale<scalar, ValueType>, ValueType>, ValueType>
    operator*(scalar s, Expr<T, ValueType> x)
    {
        typedef ScalingExpr<scalar, Expr<T, ValueType>, Scale<scalar, ValueType>, ValueType> sexp;
        return Expr<sexp, ValueType>(sexp( s, x ));
    }

    template <Scalar scalar, typename T, typename ValueType>
    Expr<ScalingExpr<scalar, Expr<T, ValueType>, Scale<scalar, ValueType>, ValueType>, ValueType>
    operator*(Expr<T, ValueType> x, scalar s)
    {
        typedef ScalingExpr<scalar, Expr<T, ValueType>, Scale<scalar, ValueType>, ValueType> sexp;
        return Expr<sexp, ValueType>(sexp( s, x ));
    }

    template <std::floating_point real>
    class solution_wrapper
    {
    public:
        typedef SparseData<matrix<real>> _umap;

        _umap values;

        solution_wrapper() {}

        solution_wrapper(const solution_wrapper<real>& a) : values(a.values) {}
        solution_wrapper<real>& operator=(const solution_wrapper<real>& a)
        {
            values = a.values;
            return *this;
        }

        solution_wrapper(solution_wrapper<real>&& a) : values(std::move(a.values)) {}
        solution_wrapper<real>& operator=(solution_wrapper<real>&& a)
        {
            values = std::move(a.values);
            return *this;
        }

        // move map
        solution_wrapper(_umap&& a) : values(std::move(a)) {}
        solution_wrapper<real>& operator=(_umap&& a)
        {
            values = std::move(a);
            return *this;
        }

        // copy map
        solution_wrapper(const _umap& a) : values(a) {}
        solution_wrapper<real>& operator=(const _umap& a)
        {
            values = a;
            return *this;
        }

        class sol_iterator
        {
        public:
            typedef std::pair<int, matrix<real>> pT;

            _umap::const_iterator it;
            size_t _size;

            inline pT operator*() const
            {
                return *it;
            }

            void operator++()
            {
                ++it;
            }

            inline size_t size() const
            {
                return _size;
            }
        };

        sol_iterator begin() const
        {
            sol_iterator x;
            x._size = values.size();
            x.it = values.cbegin();

            return x;
        }

        template <typename T>
        solution_wrapper(Expr<T, matrix<real>> x)
        {
            for (int i=0; i < x.size(); ++i)
            {
                auto xi = *x;
                values[xi.first] = xi.second;
                ++x;
            }
        }
        template <typename T>
        solution_wrapper& operator=(Expr<T, matrix<real>> x)
        {
            values.clear();
            for (int i=0; i < x.size(); ++i)
            {
                auto xi = *x;
                values[xi.first] = xi.second;
                ++x;
            }

            return *this;
        }
        template <typename T>
        solution_wrapper& operator+=(Expr<T, matrix<real>> x)
        {
            for (int i=0; i < x.size(); ++i)
            {
                auto xi = *x;
                // values[xi.first] += xi.second;
                values.at(xi.first) += xi.second;
                ++x;
            }

            return *this;
        }
        template <typename T>
        solution_wrapper& operator-=(Expr<T, matrix<real>> x)
        {
            for (int i=0; i < x.size(); ++i)
            {
                auto xi = *x;
                // values[xi.first] -= xi.second;
                values.at(xi.first) -= xi.second;
                ++x;
            }

            return *this;
        }

        solution_wrapper& operator+=(const solution_wrapper<real>& x)
        {
            for (int i=0; i < x.size(); ++i)
            {
                auto xi = *x;
                // values[xi.first] += xi.second;
                values.at(xi.first) += xi.second;
                ++x;
            }

            return *this;
        }
        solution_wrapper& operator-=(const solution_wrapper<real>& x)
        {
            for (int i=0; i < x.size(); ++i)
            {
                auto xi = *x;
                // values[xi.first] -= xi.second;
                values.at(xi.first) -= xi.second;
                ++x;
            }

            return *this;
        }
    };

    // expr + sol
    template <typename T, std::floating_point real>
    Expr<BinaryExpr<Expr<T, matrix<real>>, typename solution_wrapper<real>::sol_iterator, Plus<matrix<real>>, matrix<real>>, matrix<real>>
    operator+(const Expr<T, matrix<real>>& x, const solution_wrapper<real>& y)
    {
        typedef BinaryExpr<Expr<T, matrix<real>>, typename solution_wrapper<real>::sol_iterator, Plus<matrix<real>>, matrix<real>> bexp;
        return Expr<bexp, matrix<real>>(bexp(x, y.begin()));
    }

    // sol + expr
    template <typename T, std::floating_point real>
    Expr<BinaryExpr<typename solution_wrapper<real>::sol_iterator, Expr<T, matrix<real>>, Plus<matrix<real>>, matrix<real>>, matrix<real>>
    operator+(const solution_wrapper<real>& x, const Expr<T, matrix<real>>& y)
    {
        typedef BinaryExpr<typename solution_wrapper<real>::sol_iterator, Expr<T, matrix<real>>, Plus<matrix<real>>, matrix<real>> bexp;
        return Expr<bexp, matrix<real>>(bexp(x.begin(), y));
    }

    // sol + sol
    template <std::floating_point real>
    Expr<BinaryExpr<typename solution_wrapper<real>::sol_iterator, typename solution_wrapper<real>::sol_iterator, Plus<matrix<real>>, matrix<real>>, matrix<real>>
    operator+(const solution_wrapper<real>& x, const solution_wrapper<real>& y)
    {
        typedef BinaryExpr<typename solution_wrapper<real>::sol_iterator, typename solution_wrapper<real>::sol_iterator, Plus<matrix<real>>, matrix<real>> bexp;
        return Expr<bexp, matrix<real>>(bexp(x.begin(), y.begin()));
    }

    // Expr - sol
    template <typename T, std::floating_point real>
    Expr<BinaryExpr<Expr<T, matrix<real>>, typename solution_wrapper<real>::sol_iterator, Minus<matrix<real>>, matrix<real>>, matrix<real>>
    operator-(const Expr<T, matrix<real>>& x, const solution_wrapper<real>& y)
    {
        typedef BinaryExpr<Expr<T, matrix<real>>, typename solution_wrapper<real>::sol_iterator, Minus<matrix<real>>, matrix<real>> bexp;
        return Expr<bexp, matrix<real>>(bexp(x, y.begin()));
    }

    // sol - expr
    template <typename T, std::floating_point real>
    Expr<BinaryExpr<typename solution_wrapper<real>::sol_iterator, Expr<T, matrix<real>>, Minus<matrix<real>>, matrix<real>>, matrix<real>>
    operator-(const solution_wrapper<real>& x, const Expr<T, matrix<real>>& y)
    {
        typedef BinaryExpr<typename solution_wrapper<real>::sol_iterator, Expr<T, matrix<real>>, Minus<matrix<real>>, matrix<real>> bexp;
        return Expr<bexp, matrix<real>>(bexp(x.begin(), y));
    }

    // sol - sol
    template <std::floating_point real>
    Expr<BinaryExpr<typename solution_wrapper<real>::sol_iterator, typename solution_wrapper<real>::sol_iterator, Minus<matrix<real>>, matrix<real>>, matrix<real>>
    operator-(const solution_wrapper<real>& x, const solution_wrapper<real>& y)
    {
        typedef BinaryExpr<typename solution_wrapper<real>::sol_iterator, typename solution_wrapper<real>::sol_iterator, Minus<matrix<real>>, matrix<real>> bexp;
        return Expr<bexp, matrix<real>>(bexp(x.begin(), y.begin()));
    }

    // scalar * sol
    template <Scalar scalar, std::floating_point real>
    Expr<ScalingExpr<scalar, typename solution_wrapper<real>::sol_iterator, Scale<scalar, matrix<real>>, matrix<real>>, matrix<real>>
    operator*(scalar s, const solution_wrapper<real>& x)
    {
        typedef ScalingExpr<scalar, typename solution_wrapper<real>::sol_iterator, Scale<scalar, matrix<real>>, matrix<real>> sexp;
        return Expr<sexp, matrix<real>>(sexp(s, x.begin()));
    }

    // sol * scalar
    template <Scalar scalar, std::floating_point real>
    Expr<ScalingExpr<scalar, typename solution_wrapper<real>::sol_iterator, Scale<scalar, matrix<real>>, matrix<real>>, matrix<real>>
    operator*(const solution_wrapper<real>& x, scalar s)
    {
        typedef ScalingExpr<scalar, typename solution_wrapper<real>::sol_iterator, Scale<scalar, matrix<real>>, matrix<real>> sexp;
        return Expr<sexp, matrix<real>>(sexp(s, x.begin()));
    }

}; // namespace schro_mpi

#endif