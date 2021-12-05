#ifndef OMP_IMPL_V_WRAPPER_HPP
#define OMP_IMPL_V_WRAPPER_HPP

#include "openmp_impl/solution_wrapper.hpp"

namespace schro_omp
{
    template <std::floating_point real>
    class v_wrapper : public Expr<v_wrapper<real>, solution_wrapper<real>>
    {
    public:
        typedef real value_type;

        std::vector<solution_wrapper<real>> values;

        inline const solution_wrapper<real>& operator[](size_t i) const
        {
            return values[i];
        }

        inline solution_wrapper<real>& operator[](size_t i)
        {
            return values[i];
        }

        inline size_t size() const
        {
            return values.size();
        }

        v_wrapper() {}
        
        v_wrapper(v_wrapper<real>&& a) : values(std::move(a.values)) {}
        v_wrapper<real>& operator=(v_wrapper<real>&& a)
        {
            values = std::move(a.values);
            return *this;
        }

        v_wrapper(const v_wrapper<real>& a) : values(a.values) {}
        v_wrapper<real>& operator=(const v_wrapper<real>& a)
        {
            values = a.values;
            return *this;
        }

        template <typename T>
        v_wrapper(const Expr<T, solution_wrapper<real>>& op) : values(op.size())
        {
            for (int i=0; i < op.size(); ++i)
                values[i] = op[i];
        }
        template <typename T>
        v_wrapper<real>& operator=(const Expr<T, solution_wrapper<real>>& op)
        {
            values.resize(op.size());
            for (int i=0; i < op.size(); ++i)
                values[i] = op[i];

            return *this;
        }

        template <typename T>
        v_wrapper<real>& operator+=(const Expr<T, solution_wrapper<real>>& op)
        {
            for (int i=0; i < op.size(); ++i)
                values[i] += op[i];

            return *this;
        }

        template <typename T>
        v_wrapper<real>& operator-=(const Expr<T, solution_wrapper<real>>& op)
        {
            for (int i=0; i < op.size(); ++i)
                values[i] -= op[i];

            return *this;
        }
    };
} // namespace schro_omp


#endif