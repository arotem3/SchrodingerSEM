#ifndef OMP_IMPL_SCHRODINGER_HPP
#define OMP_IMPL_SCHRODINGER_HPP

#include <concepts>
#include <deque>

#include "types.hpp"
#include "minres.hpp"
#include "verbose.hpp"
#include "ode_base.hpp"
#include "openmp_impl/galerkin.hpp"
#include "openmp_impl/v_wrapper.hpp"
#include "openmp_impl/dot.hpp"

namespace schro_omp
{
    // solves schrodinger equation until time T. uses second order backwards
    // differetiation formula:
    // see pg 365 in:
    // Hairer, E, S P. Nørsett, and Gerhard Wanner. "Solving Ordinary
    // Differential Equations." (1993). Print.
    template <std::floating_point real, std::invocable<std::vector<matrix<real>>, std::vector<matrix<real>>> Callback = NullCallback>
    bool schrodinger_bdf2(std::vector<matrix<real>>& psi_real, std::vector<matrix<real>>& psi_imag, const std::vector<matrix<real>>& potential, const Mesh<real>& mesh, real T, const solver_opts<real>& opts, Callback callback = NullCallback{})
    {
        const int n = mesh.elements.size();
        real alpha = opts.dt;

        VerboseTracker vtracker(std::cout, 2-std::log10(opts.dt));

        v_wrapper<real> p;
        p.values.resize(2);
        p[0] = solution_wrapper<real>(std::move(psi_real));
        p[1] = solution_wrapper<real>(std::move(psi_imag));

        const auto& w = mesh.quadrature.w;

        auto L = [&](const matrix<real>& u, int i) -> matrix<real>
        {
            matrix<real> proj_Vu = (arma::diagmat(w) * (u % potential[i]) * arma::diagmat(w)) % mesh.elements[i].J;
            matrix<real> nlap = glaplace(u, mesh.elements[i], mesh.D, mesh.quadrature);

            return nlap + proj_Vu;
        }; // schrodinger operator, d^2/dx^2 + d^2/dy^2 + V(x,y)

        auto Res = [&](const v_wrapper<real>& q) -> v_wrapper<real>
        {
            v_wrapper<real> y = q;

            unmask(y[0].values, mesh); unmask(y[1].values, mesh);

            #pragma omp parallel for if(n > 63) schedule(static)
            for (int i=0; i < n; ++i)
            {
                matrix<real> u = std::move(y[0].values[i]);
                matrix<real> v = std::move(y[1].values[i]);
                matrix<real> proj_u = (arma::diagmat(w) * u * arma::diagmat(w)) % mesh.elements[i].J;
                matrix<real> proj_v = (arma::diagmat(w) * v * arma::diagmat(w)) % mesh.elements[i].J;
                
                y[0].values[i] = proj_u + alpha*L(v, i);
                y[1].values[i] = alpha*L(u, i) - proj_v;
            }

            global_sum(y[0].values, mesh); global_sum(y[1].values, mesh);
            mask(y[0].values, mesh); mask(y[1].values, mesh);

            return y;                        
        }; // I - alpha*A corresponding to residual

        auto proj = [&mesh](const v_wrapper<real>& p) -> v_wrapper<real>
        {
            v_wrapper<real> out;
            out.values.resize(2);
            out[0] = gproj<real>(p[0].values, mesh);
            out[1] = gproj<real>(p[1].values, mesh);
            out[1] = real(-1.0f) * out[1];

            return out;
        };

        auto dotprod = [&mesh](const v_wrapper<real>& p1, const v_wrapper<real>& p2) -> real
        {
            return dot<real>(mesh, p1[0].values, p2[0].values) + dot<real>(mesh, p1[1].values, p2[1].values);
        };

        std::deque<v_wrapper<real>> ps;

        bool success = true;
        real t = 0;
        while (t < T)
        {
            if (ps.size() == 2)
                ps.pop_front();
            ps.push_back(proj(p));

            solver_results<real> rslts;
            if (ps.size() == 1) {
                rslts = minres<real>(p, Res, ps[0], dotprod, IdentityPreconditioner{}, opts.max_iter, opts.tol);
                alpha = (2.0/3.0) * opts.dt;
            } else {
                v_wrapper<real> b = real(-1.0/3.0)*ps[0] + real(4.0/3.0)*ps[1];
                rslts = minres<real>(p, Res, b, dotprod, IdentityPreconditioner{}, opts.max_iter, opts.tol);
            }

            unmask(p[0].values, mesh);
            unmask(p[1].values, mesh);

            t += opts.dt;

            if (opts.verbose)
                vtracker(t, T);

            success = rslts.success;
            if (not success)
                break;

            bool please_stop = callback(p[0].values, p[1].values);
            if (please_stop)
                break;
        }

        if (opts.verbose)
            std::cout << std::endl;

        psi_real = std::move(p[0].values);
        psi_imag = std::move(p[1].values);

        return success;
    }

    // solves schrodinger equation until time T using 4th order 5-stage SDIRK
    // method. This method benefits from L-stability and has a dense
    // continuation formula.
    // see pg 107 in:
    // Hairer E., Wanner G. (1991) Stiff Problems — One-Step Methods. In:
    // Solving Ordinary Differential Equations II. Springer Series in
    // Computational Mathematics, vol 14. Springer, Berlin, Heidelberg.
    // https://doi-org.proxy2.cl.msu.edu/10.1007/978-3-662-09947-6_1
    template <std::floating_point real, std::invocable<std::vector<matrix<real>>, std::vector<matrix<real>>> Callback = NullCallback>
    bool schrodinger_dirk4s5(std::vector<matrix<real>>& psi_real, std::vector<matrix<real>>& psi_imag, const std::vector<matrix<real>>& potential, const Mesh<real>& mesh, real T, const solver_opts<real>& opts, Callback callback = NullCallback{})
    {
        const int n = mesh.elements.size();
        const real alpha = 0.25 * opts.dt;

        VerboseTracker vtracker(std::cout, 2-std::log10(opts.dt));

        v_wrapper<real> p;
        p.values.resize(2);
        p[0] = solution_wrapper<real>(std::move(psi_real));
        p[1] = solution_wrapper<real>(std::move(psi_imag));

        v_wrapper<real> zeros = real(0) * p;

        const auto& w = mesh.quadrature.w;

        auto L = [&](const matrix<real>& u, int i) -> matrix<real>
        {
            matrix<real> proj_Vu = (arma::diagmat(w) * (u % potential[i]) * arma::diagmat(w)) % mesh.elements[i].J;
            matrix<real> nlap = glaplace(u, mesh.elements[i], mesh.D, mesh.quadrature);

            return nlap + proj_Vu;
        }; // schrodinger operator, d^2/dx^2 + d^2/dy^2 + V(x,y)

        auto f = [&](const v_wrapper<real>& q) -> v_wrapper<real>
        {
            v_wrapper<real> y = q;

            unmask(y[0].values, mesh); unmask(y[1].values, mesh);

            #pragma omp parallel for if(n > 63) schedule(static)
            for (int i=0; i < n; ++i)
            {
                matrix<real> u = std::move(y[0].values[i]);
                matrix<real> v = std::move(y[1].values[i]);
                
                y[0].values[i] = -L(v, i);
                y[1].values[i] = -L(u, i);
            }

            global_sum(y[0].values, mesh); global_sum(y[1].values, mesh);
            mask(y[0].values, mesh); mask(y[1].values, mesh);

            return y;
        }; // schrodinger operator

        auto Res = [&](const v_wrapper<real>& q) -> v_wrapper<real>
        {
            v_wrapper<real> y = q;

            unmask(y[0].values, mesh); unmask(y[1].values, mesh);

            #pragma omp parallel for if(n > 63) schedule(static)
            for (int i=0; i < n; ++i)
            {
                matrix<real> u = std::move(y[0].values[i]);
                matrix<real> v = std::move(y[1].values[i]);
                matrix<real> proj_u = (arma::diagmat(w) * u * arma::diagmat(w)) % mesh.elements[i].J;
                matrix<real> proj_v = (arma::diagmat(w) * v * arma::diagmat(w)) % mesh.elements[i].J;
                
                y[0].values[i] = proj_u + alpha*L(v, i);
                y[1].values[i] = alpha*L(u, i) - proj_v;
            }

            global_sum(y[0].values, mesh); global_sum(y[1].values, mesh);
            mask(y[0].values, mesh); mask(y[1].values, mesh);

            return y;                        
        }; // I - alpha*A corresponding to residual calculcations in rk step
    
        auto dotprod = [&](const v_wrapper<real>& q1, const v_wrapper<real>& q2) -> real
        {
            return dot<real>(mesh, q1[0].values, q2[0].values) + dot<real>(mesh, q1[1].values, q2[1].values);
        };

        real pnorm = std::sqrt(dotprod(p, p));

        real t = 0;
        bool success;

        while (t < T)
        {
            v_wrapper<real> v = opts.dt * f(p);
            v_wrapper<real> z1 = zeros;
            solver_results<real> rslts = minres<real>(z1, Res, v, dotprod, IdentityPreconditioner{}, opts.max_iter, opts.tol);
            success = rslts.success;
            if (not success)
                break;

            v = p + real(0.5)*z1;
            v = opts.dt * f(v);
            v_wrapper<real> z2 = zeros;
            rslts = minres<real>(z2, Res, v, dotprod, IdentityPreconditioner{}, opts.max_iter, opts.tol);
            success = rslts.success;
            if (not success)
                break;

            v = p + real(17.0/50.0)*z1 + real(-1.0/25.0)*z2;
            v = opts.dt * f(v);
            v_wrapper<real> z3 = zeros;
            rslts = minres<real>(z3, Res, v, dotprod, IdentityPreconditioner{}, opts.max_iter, opts.tol);
            success = rslts.success;
            if (not success)
                break;

            v = p + real(371.0/1360.0)*z1 + real(-137.0/2720.0)*z2 + real(15.0/544.0)*z3;
            v = opts.dt * f(v);
            v_wrapper<real> z4 = zeros;
            rslts = minres<real>(z4, Res, v, dotprod, IdentityPreconditioner{}, opts.max_iter, opts.tol);
            success = rslts.success;
            if (not success)
                break;

            v = p + real(25.0/24.0)*z1 + real(-49.0/48.0)*z2 + real(125.0/16.0)*z3 + real(-85.0/12.0)*z4;
            v = opts.dt * f(v);
            v_wrapper<real> z5 = zeros;
            rslts = minres<real>(z5, Res, v, dotprod, IdentityPreconditioner{}, opts.max_iter, opts.tol);
            success = rslts.success;
            if (not success)
                break;

            p += real(25.0/24.0)*z1 + real(-49.0/48.0)*z2 + real(125.0/16.0)*z3 + real(-85.0/12.0)*z4 + real(0.25)*z5;

            v_wrapper<real> err = real(-3.0/16.0)*z1 + real(-27.0/32.0)*z2 + real(25.0/32.0)*z3 + real(0.25)*z5;
            real estimated_error = std::sqrt(dotprod(err, err)) / pnorm;

            unmask(p[0].values, mesh);
            unmask(p[1].values, mesh);

            t += opts.dt;

            if (opts.verbose)
                vtracker(t, T, estimated_error);

            bool please_stop = callback(p[0].values, p[1].values);
            if (please_stop)
                break;
        }

        if (opts.verbose)
            std::cout << "\n";

        psi_real = std::move(p[0].values);
        psi_imag = std::move(p[1].values);

        return success;
    }

    // solves schrodinger equation until time T using 4th order 3-stage SDIRK
    // method. This method is A-stable, seems to be much faster than dirk4s5.
    // see pg 108 in:
    // Hairer E., Wanner G. (1991) Stiff Problems — One-Step Methods. In:
    // Solving Ordinary Differential Equations II. Springer Series in
    // Computational Mathematics, vol 14. Springer, Berlin, Heidelberg.
    // https://doi-org.proxy2.cl.msu.edu/10.1007/978-3-662-09947-6_1
    template <std::floating_point real, std::invocable<std::vector<matrix<real>>, std::vector<matrix<real>>> Callback = NullCallback>
    bool schrodinger_dirk4s3(std::vector<matrix<real>>& psi_real, std::vector<matrix<real>>& psi_imag, const std::vector<matrix<real>>& potential, const Mesh<real>& mesh, real T, const solver_opts<real>& opts, Callback callback = NullCallback{})
    {
        constexpr real gamma = std::cos(M_PI/18.0) / std::sqrt(3.0) + 0.5;
        constexpr real delta = 1.0 / 6.0 / std::pow(2*gamma - 1, 2);

        const int n = mesh.elements.size();
        const real alpha = gamma * opts.dt;

        VerboseTracker vtracker(std::cout, 2-std::log10(opts.dt));

        v_wrapper<real> p;
        p.values.resize(2);
        p[0] = solution_wrapper<real>(std::move(psi_real));
        p[1] = solution_wrapper<real>(std::move(psi_imag));

        v_wrapper<real> zeros = 0 * p;

        const auto& w = mesh.quadrature.w;

        auto L = [&](const matrix<real>& u, int i) -> matrix<real>
        {
            matrix<real> proj_Vu = (arma::diagmat(w) * (u % potential[i]) * arma::diagmat(w)) % mesh.elements[i].J;
            matrix<real> nlap = glaplace(u, mesh.elements[i], mesh.D, mesh.quadrature);

            return nlap + proj_Vu;
        }; // schrodinger operator, d^2/dx^2 + d^2/dy^2 + V(x,y)

        auto f = [&](const v_wrapper<real>& q) -> v_wrapper<real>
        {
            v_wrapper<real> y = q;

            unmask(y[0].values, mesh); unmask(y[1].values, mesh);

            #pragma omp parallel for if(n > 63) schedule(static)
            for (int i=0; i < n; ++i)
            {
                matrix<real> u = std::move(y[0].values[i]);
                matrix<real> v = std::move(y[1].values[i]);
                
                y[0].values[i] = -L(v, i);
                y[1].values[i] = -L(u, i);
            }

            global_sum(y[0].values, mesh); global_sum(y[1].values, mesh);
            mask(y[0].values, mesh); mask(y[1].values, mesh);

            return y;
        }; // schrodinger operator

        auto Res = [&](const v_wrapper<real>& q) -> v_wrapper<real>
        {
            v_wrapper<real> y = q;

            unmask(y[0].values, mesh); unmask(y[1].values, mesh);

            #pragma omp parallel for if(n > 63) schedule(static)
            for (int i=0; i < n; ++i)
            {
                matrix<real> u = std::move(y[0].values[i]);
                matrix<real> v = std::move(y[1].values[i]);
                matrix<real> proj_u = (arma::diagmat(w) * u * arma::diagmat(w)) % mesh.elements[i].J;
                matrix<real> proj_v = (arma::diagmat(w) * v * arma::diagmat(w)) % mesh.elements[i].J;
                
                y[0].values[i] = proj_u + alpha*L(v, i);
                y[1].values[i] = alpha*L(u, i) - proj_v;
            }

            global_sum(y[0].values, mesh); global_sum(y[1].values, mesh);
            mask(y[0].values, mesh); mask(y[1].values, mesh);

            return y;                        
        }; // I - alpha*A corresponding to residual calculcations in rk step
    
        auto dotprod = [&](const v_wrapper<real>& q1, const v_wrapper<real>& q2) -> real
        {
            return dot<real>(mesh, q1[0].values, q2[0].values) + dot<real>(mesh, q1[1].values, q2[1].values);
        };

        bool success = true;
        real t = 0;

        while (t < T)
        {
            v_wrapper<real> v = opts.dt * f(p);
            v_wrapper<real> z1 = zeros;
            solver_results<real> rslts = minres<real>(z1, Res, v, dotprod, IdentityPreconditioner{}, opts.max_iter, opts.tol);
            success = rslts.success;
            if (not success)
                break;

            v = p + real(0.5-gamma)*z1;
            v = opts.dt * f(v);
            v_wrapper<real> z2 = zeros;
            rslts = minres<real>(z2, Res, v, dotprod, IdentityPreconditioner{}, opts.max_iter, opts.tol);
            success = rslts.success;
            if (not success)
                break;
                
            v = p + real(2*gamma)*z1 + real(1-4*gamma)*z2;
            v = opts.dt * f(v);
            v_wrapper<real> z3 = zeros;
            rslts = minres<real>(z3, Res, v, dotprod, IdentityPreconditioner{}, opts.max_iter, opts.tol);
            success = rslts.success;
            if (not success)
                break;
                
            p += delta*z1 + real(1-2*delta)*z2 + delta*z3;

            unmask(p[0].values, mesh);
            unmask(p[1].values, mesh);

            t += opts.dt;

            if (opts.verbose)
                vtracker(t, T);

            bool please_stop = callback(p[0].values, p[1].values);
            if (please_stop)
                break;
        }

        if (opts.verbose)
            std::cout << std::endl;

        psi_real = std::move(p[0].values);
        psi_imag = std::move(p[1].values);

        return success;
    }
} // namespace schro_omp


#endif