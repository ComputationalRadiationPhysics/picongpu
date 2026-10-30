/* Copyright 2014-2026 Alexander Debus and LLM agent
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pmacc/attribute/unroll.hpp>
#include <pmacc/math/Complex.hpp>
#include <pmacc/types.hpp>

#include <array>
#include <cstdint>

namespace picongpu::templates::twtstight::detail
{
    /** Truncated asymptotic correction S_n(z) to the modified Bessel function I_n(z).
     *
     * DLMF 10.40.1 gives I_n(z) ~ exp(z) / sqrt(2*pi*z) * S_n(z).
     * With A_m(n) denoting the coefficients called a_m(n) in DLMF 10.17.1,
     *
     *   A_0(n) = 1,
     *   A_m(n) = product_{j=1}^m (4*n*n - (2*j-1)^2) / (m! * 8^m),
     *   t_m(z) = (-1)^m * A_m(n) / z^m,
     *   S_n(z) is approximated here by sum_{m=0}^{12} t_m(z).
     *
     * Dividing consecutive terms yields the recurrence used below:
     *
     *   t_0 = 1,
     *   t_m = t_{m-1} * ((2*m-1)^2 - 4*n*n) / (8*m*z).
     *
     * For checking signs: S_0(z) = 1 + 1/(8*z) + 9/(128*z*z) + ...,
     * whereas S_1(z) = 1 - 3/(8*z) - 15/(128*z*z) - ... .
     * This is an asymptotic expansion, not a convergent Taylor series.
     * We retain t_0 through t_12 (thirteen terms). At |z|=50 the first omitted
     * term has magnitude about 1.50e-18 for n=0 and 1.62e-18 for n=1; this
     * motivates twelve corrections for double precision at the chosen threshold.
     * This term estimate is not a general remainder bound; see DLMF 10.40(iii).
     * The fixed bound permits unrolling on device and also supports float.
     *
     * References (NIST Digital Library of Mathematical Functions):
     * - I_n asymptotics: https://dlmf.nist.gov/10.40.E1
     * - Coefficients: https://dlmf.nist.gov/10.17.E1
     * - Remainder bounds: https://dlmf.nist.gov/10.40.iii
     *
     * @tparam T_order Bessel order, used here only for n=0 and n=1.
     * @param z Real or complex argument; callers use Re(z) >= 50.
     */
    template<uint32_t T_order, typename T_Value>
    HDINLINE T_Value besselISeries(T_Value const& z)
    {
        T_Value term(1);
        T_Value sum(1);
        T_Value const inverseZ = T_Value(1) / z;
        PMACC_UNROLL(12)
        for(int32_t m = 1; m <= 12; ++m)
        {
            int32_t const odd = 2 * m - 1;
            term *= (T_Value(odd * odd - int32_t(4u * T_order * T_order)) / T_Value(8 * m)) * inverseZ;
            sum += term;
        }
        return sum;
    }

    /** Evaluate J_0(q)/I_0(a) and J_1(q)/I_0(a) without overflowing either Bessel function.
     *
     * TWTSTight::defineCommonHelperVariables() supplies the dimensionless arguments
     *
     *   q = k*rho*sin(phi),    a = k*k*w0*w0*sin(phi)/2,
     *   rho = sqrt(Xm*Xm + x*x),    Xm = -z_pos - i*k*w0*w0/2,
     *
     * where k is the wave number, w0 the waist, and x and z_pos spatial coordinates.
     * Lengths and k must use consistent units. The formula uses sin(phi) > 0, hence
     * a is positive: the opposite TWTS laser beam with -phi is obtained by rotating the geometry.
     * Near the beam axis |Im(q)| is close to a. Thus J_n(q) and I_0(a) can each
     * overflow even though their ratio is representable.
     *
     * Rotation to modified Bessel functions (DLMF 10.27.6):
     * With s = sign(Im(q)) and u = -i s q, J_n(q) = (i s)^n I_n(u).
     * This identity is exact for the integer orders n=0,1. The sign is position
     * dependent and ensures Re(u) = |Im(q)| > 0 in the asymptotic branch.
     * Inserting the I_n expansion documented in besselISeries() gives
     *
     *   R_n = J_n(q)/I_0(a)
     *       ~ (i s)^n * exp(u-a) * sqrt(a/u) * S_n(u)/S_0(a).
     *
     * The common exponential is evaluated as exp(u-a), so exp(u) and exp(a)
     * are never formed separately. With a>0 and Re(u)>0, principal square roots
     * satisfy sqrt(a)/sqrt(u) = sqrt(a/u), without an additional branch sign.
     * The order-one ratio needs the factor i*s; the order-zero ratio does not.
     *
     * Branches and threshold:
     * - a < 50: preserve the direct PMacc Bessel calculation. For these TWTS
     *   arguments |Im(q)| <= a in exact arithmetic, so both Bessel values are safe.
     * - a >= 50 and |Im(q)| >= 50: use the combined asymptotic ratio above.
     *   DLMF 10.40.1 applies in sectors |arg(u)| <= pi/2-delta, delta>0.
     *   Here arg(u) is the complex phase in radians, and delta is an arbitrary
     *   fixed positive angular margin keeping the sector away from the imaginary
     *   axis. It belongs to the mathematical statement, not to the implementation;
     *   the code instead checks Re(u) = |Im(q)| >= 50.
     *   DLMF 10.40.5 also displays the exp(-u) contribution relevant near the
     *   imaginary axis. Its exponential magnitude relative to exp(u) is
     *   exp(-2*Re(u)) <= exp(-100), well below double precision roundoff here.
     * - a >= 50 and |Im(q)| < 50: large a alone does not justify approximating
     *   I_n(u). This includes q=0 and real q, where both oscillatory contributions
     *   to J_n matter. Evaluate the bounded J_n(q) directly and use
     *
     *     R_n ~ [J_n(q)*exp(-50)] * [exp(50-a)*sqrt(2*pi*a)/S_0(a)].
     *
     *   The split exponential avoids prematurely underflowing exp(-a), while
     *   keeping the numerator within range even in float precision.
     * The threshold and series length target truncation below rounding error;
     * they do not remove rounding already present in the input arguments u and a.
     * Reference comparisons are in unit/dimensional_tests/TWTSTight.cpp.
     *
     * References (NIST Digital Library of Mathematical Functions):
     * - Exact J_n/I_n connection: https://dlmf.nist.gov/10.27.E6
     * - Dominant exponential expansion: https://dlmf.nist.gov/10.40.E1
     * - Expansion including both exponentials: https://dlmf.nist.gov/10.40.E5
     *
     * @param q Complex TWTS Bessel argument.
     * @param a Positive real normalization argument.
     * @return {J_0(q)/I_0(a), J_1(q)/I_0(a)}.
     */
    template<typename T_Float>
    HDINLINE std::array<alpaka::Complex<T_Float>, 2u> besselJOverI0(alpaka::Complex<T_Float> const& q, T_Float const a)
    {
        namespace math = pmacc::math;
        using Complex = alpaka::Complex<T_Float>;
        constexpr T_Float threshold = T_Float(50);
        if(a < threshold)
        {
            T_Float const denominator = math::bessel::i0(a);
            return {math::bessel::j0(q) / denominator, math::bessel::j1(q) / denominator};
        }

        T_Float const denominatorSeries = besselISeries<0u>(a);
        if(math::abs(q.imag()) < threshold)
        {
            // Far from the axis q can be small or real even for large a. Keep both
            // oscillatory contributions to J_n there, and scale before multiplying.
            T_Float const scale = math::exp(threshold - a)
                                  * (math::sqrt(T_Float(2) * math::Pi<T_Float>::value * a) / denominatorSeries);
            T_Float const numeratorScale = math::exp(-threshold);
            return {(math::bessel::j0(q) * numeratorScale) * scale, (math::bessel::j1(q) * numeratorScale) * scale};
        }

        T_Float const sign = q.imag() < T_Float(0) ? T_Float(-1) : T_Float(1);
        Complex const is(0, sign);
        Complex const u = -is * q;
        Complex const scale = math::exp(u - a) * math::sqrt(a / u) / denominatorSeries;
        return {scale * besselISeries<0u>(u), is * scale * besselISeries<1u>(u)};
    }
} // namespace picongpu::templates::twtstight::detail
