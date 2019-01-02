/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera,
 *                     Alexander Grund
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

#include "picongpu/simulation_defines.hpp"


namespace picongpu
{
namespace particles
{
namespace manipulators
{
namespace unary
{
namespace acc
{

    /** manipulate the speed based on a temperature
     *
     * @tparam T_ParamClass picongpu::particles::manipulators::unary::param::TemperatureCfg,
     *                      type with compile configuration
     * @tparam T_ValueFunctor pmacc::nvidia::functors, binary operator type to reduce current and new value
     */
    template<
        typename T_ParamClass,
        typename T_ValueFunctor
    >
    struct Temperature : private T_ValueFunctor
    {
        /** manipulate the speed of the particle
         *
         * @tparam T_Rng pmacc::nvidia::rng::RNG, type of the random number generator
         * @tparam T_Particle pmacc::Particle, particle type
         * @tparam T_Args pmacc::Particle, arbitrary number of particles types
         *
         * @param rng random number generator
         * @param particle particle to be manipulated
         * @param ... unused particles
         */
        template<
            typename T_Rng,
            typename T_Particle,
            typename ... T_Args
        >
        HDINLINE void operator()(
            T_Rng & rng,
            T_Particle & particle,
            T_Args && ...
        )
        {
            using ParamClass = T_ParamClass;

            const float3_X tmpRand = float3_X(
                rng(),
                rng(),
                rng()
            );
            float_X const macroWeighting = particle[ weighting_ ];

            float_X const energy = ( ParamClass::temperature * UNITCONV_keV_to_Joule ) / UNIT_ENERGY;

            // since energy is related to one particle
            // and our units are normalized for macro particle quanities
            // energy is quite small
            float_X const macroEnergy = macroWeighting * energy;
            // non-rel, MW:
            //    p = m * v
            //            v ~ sqrt(k*T/m), k*T = E
            // => p = sqrt(m)
            //
            // Note on macro particle energies, with weighting w:
            //    p_1 = m_1 * v
            //                v = v_1 = v_w
            //    p_w = p_1 * w
            //    E_w = E_1 * w
            // Since masses, energies and momenta add up linear, we can
            // just take w times the p_1. Take care, E means E_1 !
            // This goes to:
            //    p_w = w * p_1 = w * m_1 * sqrt( E / m_1 )
            //        = sqrt( E * w^2 * m_1 )
            //        = sqrt( E * w * m_w )
            // Which makes sense, since it means that we use a macroMass
            // and a macroEnergy now.
            float3_X const mom = tmpRand * ( float_X )math::sqrt(
                precisionCast< sqrt_X >(
                    macroEnergy *
                    attribute::getMass(macroWeighting,particle)
                )
            );
            T_ValueFunctor::operator( )( particle[ momentum_ ], mom );
        }
    };

} // namespace acc
} // namespace unary
} // namespace manipulators
} // namespace particles
} // namespace picongpu
