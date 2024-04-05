/* Copyright 2013-2023 Heiko Burau, Rene Widera, Richard Pausch,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/algorithms/TypeCast.hpp"
#include "pmacc/algorithms/math.hpp"
#include "pmacc/math/complex/Complex.hpp"
#include "pmacc/mpi/GetMPI_StructAsArray.hpp"
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
            using result = alpaka::Complex<T_Type>;

            HDINLINE result operator()(const T_Type& magnitude, const T_Type& phase)
            {
                return result(magnitude * pmacc::math::cos(phase), magnitude * pmacc::math::sin(phase));
            }

            HDINLINE result operator()(const T_Type& magnitude, const T_Type& sinValue, const T_Type& cosValue)
            {
                return result(magnitude * cosValue, magnitude * sinValue);
            }
        };

        //! Specialize norm() for complex numbers following the C++ standard
        template<typename T_Type>
        struct Norm<alpaka::Complex<T_Type>>
        {
            using result = typename alpaka::Complex<T_Type>::value_type;

            HDINLINE result operator()(const alpaka::Complex<T_Type>& other)
            {
                return other.real() * other.real() + other.imag() * other.imag();
            }
        };

    } // namespace math
} // namespace pmacc

namespace pmacc
{
    namespace algorithms
    {
        namespace precisionCast
        {
            /*  Specialize precisionCast-operators for alpaka complex numbers. */

            template<typename T_CastToType>
            struct TypeCast<T_CastToType, alpaka::Complex<T_CastToType>>
            {
                using result = const alpaka::Complex<T_CastToType>;

                HDINLINE result operator()(const alpaka::Complex<T_CastToType>& complexNumber) const
                {
                    return complexNumber;
                }
            };

            template<typename T_CastToType, typename T_OldType>
            struct TypeCast<T_CastToType, alpaka::Complex<T_OldType>>
            {
                using result = alpaka::Complex<T_CastToType>;

                HDINLINE result operator()(const alpaka::Complex<T_OldType>& complexNumber) const
                {
                    return result(complexNumber);
                }
            };

        } // namespace precisionCast
    } // namespace algorithms

    namespace mpi
    {
        // Specialize complex type grid buffer for MPI
        template<>
        HINLINE MPI_StructAsArray getMPI_StructAsArray<alpaka::Complex<float>>()
        {
            using ComplexType = alpaka::Complex<float>;
            MPI_StructAsArray result = getMPI_StructAsArray<ComplexType::value_type>();
            result.sizeMultiplier *= uint32_t(sizeof(ComplexType) / sizeof(ComplexType::value_type));
            return result;
        };

        // Specialize complex type grid buffer for MPI
        template<>
        HINLINE MPI_StructAsArray getMPI_StructAsArray<alpaka::Complex<double>>()
        {
            using ComplexType = alpaka::Complex<double>;
            MPI_StructAsArray result = getMPI_StructAsArray<ComplexType::value_type>();
            result.sizeMultiplier *= uint32_t(sizeof(ComplexType) / sizeof(ComplexType::value_type));
            return result;
        };

    } // namespace mpi
} // namespace pmacc
