/* Copyright 2013-2024 Axel Huebl, Heiko Burau, Rene Widera,
 *                     Alexander Grund, Sergei Bastrakov, Sergey Ermakov
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

#include "picongpu/defines.hpp"
#include "picongpu/particles/functor/User.hpp"
#include "picongpu/traits/attribute/GetMass.hpp"

namespace picongpu::particles::manipulators::unary::acc
{
    namespace detail
    {
        /** Functor to modify particle momentum based on temperature for relativistic case
         * Use Maxwell Juettner Distribution, implementation based on https://doi.org/10.1063/1.4919383
         * This functor is for the highly-relativistic case.
         * In this case the added momentum follows the Maxwell-Juettner distribution.
         *
         * @tparam T_ValueFunctor pmacc::math::operation::*, binary functor type to
         *                        add a new momentum to an old one
         */
        /** USE THIS IN THE SAME WAY AS Temperature.hpp JUST REPLACE Temperature with MaxwellJuettner
         * FOR ALL FUNCTOR AND TEMPLATE NAMES**/
        template<typename T_ValueFunctor>
        struct MaxwellJuettnerImpl : private T_ValueFunctor
        {
            /** Manipulate the momentum of the given macroparticle
             *
             * @tparam T_UniformRng functor::misc::RngWrapper, standard
             *                             normal random number generator type
             * @tparam T_Particle particle type
             * @tparam T_MaxwellJuettner temperature type
             *
             * @param uniformRng random number generator
             * @param particle particle to be manipulated
             * @param temperatureKeV temperature value in keV
             */
            template<typename T_UniformNoZeroRng, typename T_Particle, typename T_MaxwellJuettner>
            HDINLINE void operator()(
                T_UniformNoZeroRng& uniformNoZeroRng,
                T_Particle& particle,
                T_MaxwellJuettner temperatureKeV) const
            {
                float_X const energy = (temperatureKeV * sim.si.conv().eV2Joule(1.0e3)) / sim.unit.energy();

                float_X const macroWeighting = particle[weighting_];
                float_X const macroMass = picongpu::traits::attribute::getMass(macroWeighting, particle);
                float_X const mass = macroMass / macroWeighting;
                float_X const theta = energy / (mass * sim.pic.getSpeedOfLight() * sim.pic.getSpeedOfLight());

                float_X x1, x2, x3, x4, u, nu;
                bool rejected = true;
                while(rejected)
                {
                    x1 = uniformNoZeroRng();
                    x2 = uniformNoZeroRng();
                    x3 = uniformNoZeroRng();

                    u = (-1._X) * theta * math::log(x1 * x2 * x3);

                    x4 = uniformNoZeroRng();
                    nu = (-1._X) * theta * math::log(x1 * x2 * x3 * x4);

                    if(nu * nu - u * u > 1._X)
                    {
                        rejected = false;
                    }
                }

                float_X momAbs = u * macroWeighting * mass;
                float_X y1 = uniformNoZeroRng();
                float_X y2 = uniformNoZeroRng();

                float_X sin_2_pi_y2;
                float_X cos_2_pi_y2;
                math::sincos(precisionCast<float_X>(2._X * PI * y2), sin_2_pi_y2, cos_2_pi_y2);

                auto mom = float3_X(
                    momAbs * (2._X * y1 - 1._X),
                    2._X * momAbs * math::sqrt(y1 * (1._X - y1)) * cos_2_pi_y2,
                    2._X * momAbs * math::sqrt(y1 * (1._X - y1)) * sin_2_pi_y2);

                T_ValueFunctor::operator()(particle[momentum_], mom);
            }
        };

    } // namespace detail

    template<typename T_ParamClass, typename T_ValueFunctor>
    struct MaxwellJuettner : public detail::MaxwellJuettnerImpl<T_ValueFunctor>
    {
        //! Base class
        using Base = detail::MaxwellJuettnerImpl<T_ValueFunctor>;

        /** Manipulate the momentum of the given macroparticle
         *
         * @tparam T_UniformRng functor::misc::RngWrapper,
         *                             uniform random number generator type
         * @tparam T_Particle particle type
         *
         * @param uniform random number generator
         * @param particle particle to be manipulated
         */
        template<typename T_UniformRng, typename T_Particle>
        HDINLINE void operator()(T_UniformRng& uniformRng, T_Particle& particle)
        {
            auto const temperatureKeV = T_ParamClass::temperature;
            Base::operator()(uniformRng, particle, temperatureKeV);
        }
    };

    template<typename T_MaxwellJuettnerFunctor, typename T_ValueFunctor>
    struct FreeMaxwellJuettner
        : public detail::MaxwellJuettnerImpl<T_ValueFunctor>
        , public particles::functor::User<T_MaxwellJuettnerFunctor>
    {
        //! Base implementation class
        using Base = detail::MaxwellJuettnerImpl<T_ValueFunctor>;

        //! Wrapper around user-provided functor
        using UserFunctor = particles::functor::User<T_MaxwellJuettnerFunctor>;

        /** Create a functor instance, including instances for user functor and its wrapper
         *
         * @param currentStep current time iteration
         */
        FreeMaxwellJuettner(uint32_t const currentStep, IdGenerator idGen) : UserFunctor(currentStep, idGen)
        {
        }

        /** Manipulate the momentum of the given macroparticle
         *
         * @tparam T_UniformRng functor::misc::RngWrapper, standard
         *                             uniform random number generator type
         * @tparam T_Particle particle type
         *
         * @param totalCellOffset total offset including all slides [in cells]
         * @param uniformRng uniform random number generator
         * @param particle particle to be manipulated
         */
        template<typename T_UniformRng, typename T_Particle>
        HDINLINE void operator()(
            DataSpace<simDim> const& totalCellOffset,
            T_UniformRng& uniformRng,
            T_Particle& particle)
        {
            auto const unitLength = sim.unit.length();
            auto const cellSize_SI = precisionCast<float_64>(sim.pic.getCellSize()) * unitLength;
            auto const position_SI
                = (precisionCast<float_64>(totalCellOffset) + precisionCast<float_64>(particle[position_]))
                * cellSize_SI.shrink<simDim>();
            auto const temperatureKeV = UserFunctor::operator()(position_SI, cellSize_SI);
            Base::operator()(uniformRng, particle, temperatureKeV);
        }
    };
} // namespace picongpu::particles::manipulators::unary::acc
