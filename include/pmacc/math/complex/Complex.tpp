/* Copyright 2013-2021 Heiko Burau, Rene Widera, Richard Pausch,
 *                     Alexander Debus, Benjamin Worpitz, Finn-Ole Carstens
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/algorithms/math.hpp"
#include "pmacc/algorithms/TypeCast.hpp"
#include "pmacc/math/Complex.hpp"

#include "pmacc/traits/GetComponentsType.hpp"
#include "pmacc/traits/GetNComponents.hpp"

#include <cmath>

namespace pmacc
{
    namespace math
    {
        /*  Set primary template and subsequent specialization for returning a complex number
            by using Euler's formula. */

        template<typename T_Type>
        struct Euler;

        template<typename T_Type>
        HDINLINE typename Euler<T_Type>::result euler(const T_Type& magnitude, const T_Type& phase)
        {
            return Euler<T_Type>()(magnitude, phase);
        }

        template<typename T_Type>
        HDINLINE typename Euler<T_Type>::result euler(
            const T_Type& magnitude,
            const T_Type& sinValue,
            const T_Type& cosValue)
        {
            return Euler<T_Type>()(magnitude, sinValue, cosValue);
        }

        template<typename T_Type>
        struct Euler
        {
            typedef typename ::pmacc::math::Complex<T_Type> result;

            HDINLINE result operator()(const T_Type& magnitude, const T_Type& phase)
            {
                return result(magnitude * cupla::math::cos(phase), magnitude * cupla::math::sin(phase));
            }

            HDINLINE result operator()(const T_Type& magnitude, const T_Type& sinValue, const T_Type& cosValue)
            {
                return result(magnitude * cosValue, magnitude * sinValue);
            }
        };

        /*  Set primary template and subsequent specialization of arg() for retrieving
         *  the phase of a complex number (Note: Branchcut running from -infinity to 0).
         */
        template<typename T_Type>
        struct Arg;

        template<typename T_Type>
        HDINLINE typename Arg<T_Type>::result arg(const T_Type& val)
        {
            return Arg<T_Type>()(val);
        }

        template<typename T_Type>
        struct Arg<::pmacc::math::Complex<T_Type>>
        {
            typedef typename ::pmacc::math::Complex<T_Type>::type result;
            typedef T_Type type;

            HDINLINE result operator()(const ::pmacc::math::Complex<T_Type>& other)
            {
                if(other.get_real() == type(0.0) && other.get_imag() == type(0.0))
                    return type(0.0);
                else if(other.get_real() == type(0.0) && other.get_imag() > type(0.0))
                    return Pi<type>::halfValue;
                else if(other.get_real() == type(0.0) && other.get_imag() < type(0.0))
                    return -Pi<type>::halfValue;
                else if(other.get_real() < type(0.0) && other.get_imag() == type(0.0))
                    return Pi<type>::value;
                else
                    return cupla::math::atan2(other.get_imag(), other.get_real());
            }
        };

        /** Specialize abs2() for complex numbers.
         *
         * Note: Abs is specialized in alpaka::math below
         */
        template<typename T_Type>
        struct Abs2<::pmacc::math::Complex<T_Type>>
        {
            typedef typename ::pmacc::math::Complex<T_Type>::type result;

            HDINLINE result operator()(const ::pmacc::math::Complex<T_Type>& other)
            {
                return pmacc::math::abs2(other.get_real()) + pmacc::math::abs2(other.get_imag());
            }
        };

    } // namespace math
} // namespace pmacc

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            template<typename T_Ctx, typename T_Type>
            struct Pow<T_Ctx, ::pmacc::math::Complex<T_Type>, T_Type, void>
            {
                ALPAKA_FN_HOST_ACC static auto pow(
                    T_Ctx const& mathConcept,
                    ::pmacc::math::Complex<T_Type> const& other,
                    T_Type const& exponent) -> ::pmacc::math::Complex<T_Type>
                {
                    return cupla::pow(cupla::math::abs(other), exponent)
                        * cupla::math::exp(
                               ::pmacc::math::Complex<T_Type>(T_Type(0.), T_Type(1.)) * pmacc::math::arg(other)
                               * exponent);
                }
            };

            template<typename T_Ctx, typename T_Type>
            struct Sqrt<T_Ctx, ::pmacc::math::Complex<T_Type>, void>
            {
                ALPAKA_FN_HOST_ACC static auto sqrt(
                    T_Ctx const& mathConcept,
                    ::pmacc::math::Complex<T_Type> const& other) -> ::pmacc::math::Complex<T_Type>
                {
                    using type = T_Type;
                    if(other.get_real() <= type(0.0) && other.get_imag() == type(0.0))
                    {
                        return ::pmacc::math::Complex<T_Type>(
                            type(0.0),
                            alpaka::math::sqrt(mathConcept, -other.get_real()));
                    }
                    else
                    {
                        return alpaka::math::sqrt(mathConcept, cupla::math::abs(other))
                            * (other + cupla::math::abs(other)) / cupla::math::abs(other + cupla::math::abs(other));
                    }
                }
            };

            template<typename T_Ctx, typename T_Type>
            struct Exp<T_Ctx, ::pmacc::math::Complex<T_Type>, void>
            {
                ALPAKA_FN_HOST_ACC static auto exp(
                    T_Ctx const& mathConcept,
                    ::pmacc::math::Complex<T_Type> const& other) -> ::pmacc::math::Complex<T_Type>
                {
                    using type = T_Type;
                    return pmacc::math::euler(type(1.0), other.get_imag())
                        * alpaka::math::exp(mathConcept, other.get_real());
                }
            };

            template<typename T_Ctx, typename T_Type>
            struct Abs<T_Ctx, ::pmacc::math::Complex<T_Type>, void>
            {
                ALPAKA_FN_HOST_ACC static auto abs(
                    T_Ctx const& mathConcept,
                    ::pmacc::math::Complex<T_Type> const& other) -> T_Type
                {
                    /* It is not possible to use alpaka::math::sqrt( mathConcept, ... )
                     * here, as the mathConcept would not match, so go around via cupla
                     */
                    return cupla::math::sqrt(pmacc::math::abs2(other));
                }
            };

            template<typename T_Ctx, typename T_Type>
            struct Log<T_Ctx, ::pmacc::math::Complex<T_Type>, void>
            {
                ALPAKA_FN_HOST_ACC static auto log(
                    T_Ctx const& mathConcept,
                    ::pmacc::math::Complex<T_Type> const& other) -> ::pmacc::math::Complex<T_Type>
                {
                    using type = T_Type;
                    return alpaka::math::log(mathConcept, cupla::math::abs(other))
                        + ::pmacc::math::Complex<T_Type>(type(0.), type(1.)) * pmacc::math::arg(other);
                }
            };

            template<typename T_Ctx, typename T_Type>
            struct Cos<T_Ctx, ::pmacc::math::Complex<T_Type>, void>
            {
                ALPAKA_FN_HOST_ACC static auto cos(
                    T_Ctx const& mathConcept,
                    ::pmacc::math::Complex<T_Type> const& other) -> ::pmacc::math::Complex<T_Type>
                {
                    using type = T_Type;
                    return (alpaka::math::exp(mathConcept, ::pmacc::math::Complex<T_Type>(type(0.), type(1.)) * other)
                            + alpaka::math::exp(
                                mathConcept,
                                ::pmacc::math::Complex<T_Type>(type(0.), type(-1.)) * other))
                        / type(2.0);
                }
            };

            template<typename T_Ctx, typename T_Type>
            struct Sin<T_Ctx, ::pmacc::math::Complex<T_Type>, void>
            {
                ALPAKA_FN_HOST_ACC static auto sin(
                    T_Ctx const& mathConcept,
                    ::pmacc::math::Complex<T_Type> const& other) -> ::pmacc::math::Complex<T_Type>
                {
                    using type = T_Type;

                    return (alpaka::math::exp(mathConcept, ::pmacc::math::Complex<T_Type>(type(0.), type(1.)) * other)
                            - alpaka::math::exp(
                                mathConcept,
                                ::pmacc::math::Complex<T_Type>(type(0.), type(-1.)) * other))
                        / ::pmacc::math::Complex<T_Type>(type(0.), type(2.));
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka


namespace pmacc
{
    namespace algorithms
    {
        namespace precisionCast
        {
            /*  Specialize precisionCast-operators for complex numbers. */

            template<typename T_CastToType>
            struct TypeCast<T_CastToType, ::pmacc::math::Complex<T_CastToType>>
            {
                typedef const ::pmacc::math::Complex<T_CastToType>& result;

                HDINLINE result operator()(const ::pmacc::math::Complex<T_CastToType>& complexNumber) const
                {
                    return complexNumber;
                }
            };

            template<typename T_CastToType, typename T_OldType>
            struct TypeCast<T_CastToType, ::pmacc::math::Complex<T_OldType>>
            {
                typedef ::pmacc::math::Complex<T_CastToType> result;

                HDINLINE result operator()(const ::pmacc::math::Complex<T_OldType>& complexNumber) const
                {
                    return result(complexNumber);
                }
            };

        } // namespace precisionCast
    } // namespace algorithms

    namespace mpi
    {
        using complex_X = pmacc::math::Complex<picongpu::float_X>;

        // Specialize complex type grid buffer for MPI
        template<>
        MPI_StructAsArray getMPI_StructAsArray<pmacc::math::Complex<picongpu::float_X>>()
        {
            MPI_StructAsArray result = getMPI_StructAsArray<complex_X::type>();
            result.sizeMultiplier *= uint32_t(sizeof(complex_X) / sizeof(typename complex_X::type));
            return result;
        };

    } // namespace mpi
} // namespace pmacc
