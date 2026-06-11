/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/traits/attribute/GetMass.hpp"

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
                    /** manipulate the speed
                     *
                     * @tparam T_ParamClass picongpu::particles::manipulators::unary::param::DriftCfg,
                     *                      type with compile configuration
                     * @tparam T_ValueFunctor pmacc::math::operation, binary operator type to reduce current and new
                     * value
                     */
                    template<typename T_ParamClass, typename T_ValueFunctor>
                    struct Drift : private T_ValueFunctor
                    {
                        /** manipulate the speed of the particle
                         *
                         * @tparam T_Particle pmacc::Particle, particle type
                         *
                         * @param particle particle to be manipulated
                         */
                        template<typename T_Particle>
                        HDINLINE void operator()(T_Particle& particle)
                        {
                            using ParamClass = T_ParamClass;
                            using ValueFunctor = T_ValueFunctor;

                            float_X const macroWeighting = particle[weighting_];
                            float_X const macroMass = picongpu::traits::attribute::getMass(macroWeighting, particle);

                            float_64 const myGamma = ParamClass::gamma;

                            float_64 const initFreeBeta = math::sqrt(1.0 - 1.0 / (myGamma * myGamma));

                            float3_X const driftDirection = ParamClass::driftDirection;
                            float3_X const normDir = driftDirection / pmacc::math::l2norm(driftDirection);

                            float3_X const mom(
                                normDir
                                * float_X(
                                    myGamma * initFreeBeta * float_64(macroMass)
                                    * float_64(sim.pic.getSpeedOfLight())));

                            ValueFunctor::operator()(particle[momentum_], mom);
                        }
                    };

                } // namespace acc
            } // namespace unary
        } // namespace manipulators
    } // namespace particles
} // namespace picongpu
