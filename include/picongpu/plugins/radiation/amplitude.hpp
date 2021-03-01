/* Copyright 2013-2021 Heiko Burau, Rene Widera, Richard Pausch, Alexander Debus
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "VectorTypes.hpp"

#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/math/Complex.hpp>
#include <pmacc/mpi/GetMPI_StructAsArray.hpp>

namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            /** class to store 3 complex numbers for the radiated amplitude
             */
            template<typename T_Float = picongpu::float_64>
            class Amplitude
            {
            public:
                /* For the intermediate amplitude values we may use single precision,
                 * for the final accumulation we will have to use double precision.
                 */
                using complex_T = pmacc::math::Complex<T_Float>;
                /* number of scalar components in Amplitude = 3 (3D) * 2 (complex) = 6 */
                static constexpr uint32_t numComponents
                    = uint32_t(3) * uint32_t(sizeof(complex_T) / sizeof(typename complex_T::type));

                /** constructor
                 *
                 * Arguments:
                 * - vector_64: real 3D vector
                 * - float: complex phase */
                DINLINE Amplitude(vector_64 vec, picongpu::float_X phase)
                {
                    picongpu::float_X cosValue;
                    picongpu::float_X sinValue;
                    pmacc::math::sincos(phase, sinValue, cosValue);
                    amp_x = pmacc::math::euler(
                        precisionCast<T_Float>(vec.x()),
                        precisionCast<T_Float>(sinValue),
                        precisionCast<T_Float>(cosValue));
                    amp_y = pmacc::math::euler(
                        precisionCast<T_Float>(vec.y()),
                        precisionCast<T_Float>(sinValue),
                        precisionCast<T_Float>(cosValue));
                    amp_z = pmacc::math::euler(
                        precisionCast<T_Float>(vec.z()),
                        precisionCast<T_Float>(sinValue),
                        precisionCast<T_Float>(cosValue));
                }

                /** default constructor
                 *
                 * \warning does not initialize values! */
                HDINLINE Amplitude(void)
                {
                }


                /** constructor
                 *
                 * Arguments:
                 * - 6x float: Re(x), Im(x), Re(y), Im(y), Re(z), Im(z) */
                HDINLINE Amplitude(
                    const picongpu::float_64 x_re,
                    const picongpu::float_64 x_im,
                    const picongpu::float_64 y_re,
                    const picongpu::float_64 y_im,
                    const picongpu::float_64 z_re,
                    const picongpu::float_64 z_im)
                    : amp_x(x_re, x_im)
                    , amp_y(y_re, y_im)
                    , amp_z(z_re, z_im)
                {
                }

                /** constructor with member initialization
                 *
                 *  @param x pmacc::math::complex x component of the amplitude vector.
                 *  @param y pmacc::math::complex y component of the amplitude vector.
                 *  @param z pmacc::math::complex z component of the amplitude vector.
                 */
                HDINLINE Amplitude(const complex_T& x, const complex_T& y, const complex_T& z)
                    : amp_x(x)
                    , amp_y(y)
                    , amp_z(z)
                {
                }

                /** returns a zero amplitude vector
                 *
                 * used to initialize amplitudes to zero */
                HDINLINE static Amplitude zero(void)
                {
                    Amplitude result;
                    result.amp_x = complex_T::zero();
                    result.amp_y = complex_T::zero();
                    result.amp_z = complex_T::zero();
                    return result;
                }

                /** assign addition */
                HDINLINE Amplitude& operator+=(const Amplitude& other)
                {
                    amp_x += other.amp_x;
                    amp_y += other.amp_y;
                    amp_z += other.amp_z;
                    return *this;
                }


                /** assign difference */
                HDINLINE Amplitude& operator-=(const Amplitude& other)
                {
                    amp_x -= other.amp_x;
                    amp_y -= other.amp_y;
                    amp_z -= other.amp_z;
                    return *this;
                }


                /** calculate radiation from *this amplitude
                 *
                 * Returns: \f$\frac{d^2 I}{d \Omega d \omega} = const*Amplitude^2\f$ */
                HDINLINE picongpu::float_64 calc_radiation(void)
                {
                    // const SI factor radiation
                    const picongpu::float_64 factor = 1.0
                        / (16. * util::cube(pmacc::math::Pi<picongpu::float_64>::value) * picongpu::EPS0
                           * picongpu::SPEED_OF_LIGHT);

                    return factor * (pmacc::math::abs2(amp_x) + pmacc::math::abs2(amp_y) + pmacc::math::abs2(amp_z));
                }


                /** debugging method
                 *
                 * Returns: real-x-value */
                HDINLINE picongpu::float_64 debug(void)
                {
                    return amp_x.get_real();
                }

                /** Getters for the components
                 */
                HDINLINE complex_T getXcomponent() const
                {
                    return this->amp_x;
                }
                HDINLINE complex_T getYcomponent() const
                {
                    return this->amp_y;
                }
                HDINLINE complex_T getZcomponent() const
                {
                    return this->amp_z;
                }

            private:
                complex_T amp_x; // complex amplitude x-component
                complex_T amp_y; // complex amplitude y-component
                complex_T amp_z; // complex amplitude z-component
            };
        } // namespace radiation
    } // namespace plugins
} // namespace picongpu

namespace pmacc
{
    namespace mpi
    {
        /** implementation of MPI transaction on Amplitude class */
        template<>
        HINLINE MPI_StructAsArray getMPI_StructAsArray<picongpu::plugins::radiation::Amplitude<>>()
        {
            MPI_StructAsArray result
                = getMPI_StructAsArray<picongpu::plugins::radiation::Amplitude<>::complex_T::type>();
            result.sizeMultiplier *= picongpu::plugins::radiation::Amplitude<>::numComponents;
            return result;
        };

    } // namespace mpi
} // namespace pmacc


namespace pmacc
{
    namespace algorithms
    {
        namespace precisionCast
        {
            /* We want to be able to cast a low
             * precision amplitude to a high-precision one.
             * The functors create temporary Amplitude objects and can
             * be detrimental to performance.
             */
            template<typename CastToType>
            struct TypeCast<CastToType, picongpu::plugins::radiation::Amplitude<CastToType>>
            {
                using result = const picongpu::plugins::radiation::Amplitude<CastToType>&;

                HDINLINE result operator()(result amplitude) const
                {
                    return amplitude;
                }
            };

            template<typename CastToType, typename OldType>
            struct TypeCast<CastToType, picongpu::plugins::radiation::Amplitude<OldType>>
            {
                using result = picongpu::plugins::radiation::Amplitude<CastToType>;
                using ParamType = picongpu::plugins::radiation::Amplitude<OldType>;
                HDINLINE result operator()(const ParamType& amplitude) const
                {
                    result Result(
                        precisionCast<result::complex_T::type>(amplitude.getXcomponent()),
                        precisionCast<result::complex_T::type>(amplitude.getYcomponent()),
                        precisionCast<result::complex_T::type>(amplitude.getZcomponent()));
                    return Result;
                }
            };

        } // namespace precisionCast
    } // namespace algorithms
} // namespace pmacc
