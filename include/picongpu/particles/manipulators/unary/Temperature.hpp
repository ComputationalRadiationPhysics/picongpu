/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera,
 *                     Alexander Grund, Sergei Bastrakov
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/functor/User.hpp"

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
                    namespace detail
                    {
                        /** Functor to modify particle momentum based on temperature
                         *
                         * This functor is for the non-relativistic case only.
                         * In this case the added momentum follows the Maxwell-Boltzmann distribution.
                         *
                         * @tparam T_ValueFunctor pmacc::math::operation::*, binary functor type to
                         *                        add a new momentum to an old one
                         */
                        template<typename T_ValueFunctor>
                        struct TemperatureImpl : private T_ValueFunctor
                        {
                            /** Manipulate the momentum of the given macroparticle
                             *
                             * @tparam T_StandardNormalRng functor::misc::RngWrapper, standard
                             *                             normal random number generator type
                             * @tparam T_Particle particle type
                             *
                             * @param standardNormalRng standard normal random number generator
                             * @param particle particle to be manipulated
                             * @param temperatureKeV temperature value in keV
                             */
                            template<typename T_StandardNormalRng, typename T_Particle, typename T_Temperature>
                            HDINLINE void operator()(
                                T_StandardNormalRng& standardNormalRng,
                                T_Particle& particle,
                                T_Temperature temperatureKeV) const
                            {
                                /* In the non-relativistic case, the added momentum follows
                                 * the Maxwell-Boltzmann distribution: each component is
                                 * independently normally distributed with zero mean and variance of
                                 * m * k * T = m * E.
                                 * For the macroweighted momentums we store as particle[ momentum_ ],
                                 * the same relation holds, just m and E are also macroweighted
                                 */
                                float_X const energy
                                    = (temperatureKeV * sim.si.conv.ev2Joule(1.0e3)) / sim.unit.energy();
                                float_X const macroWeighting = particle[weighting_];
                                float_X const macroEnergy = macroWeighting * energy;
                                float_X const macroMass = attribute::getMass(macroWeighting, particle);
                                float_X const standardDeviation
                                    = static_cast<float_X>(math::sqrt(precisionCast<sqrt_X>(macroEnergy * macroMass)));
                                float3_X const mom
                                    = float3_X(standardNormalRng(), standardNormalRng(), standardNormalRng())
                                    * standardDeviation;
                                T_ValueFunctor::operator()(particle[momentum_], mom);
                            }
                        };

                    } // namespace detail

                    template<typename T_ParamClass, typename T_ValueFunctor>
                    struct Temperature : public detail::TemperatureImpl<T_ValueFunctor>
                    {
                        //! Base class
                        using Base = detail::TemperatureImpl<T_ValueFunctor>;

                        /** Manipulate the momentum of the given macroparticle
                         *
                         * @tparam T_StandardNormalRng functor::misc::RngWrapper, standard
                         *                             normal random number generator type
                         * @tparam T_Particle particle type
                         *
                         * @param standardNormalRng standard normal random number generator
                         * @param particle particle to be manipulated
                         */
                        template<typename T_StandardNormalRng, typename T_Particle>
                        HDINLINE void operator()(T_StandardNormalRng& standardNormalRng, T_Particle& particle)
                        {
                            auto const temperatureKeV = T_ParamClass::temperature;
                            Base::operator()(standardNormalRng, particle, temperatureKeV);
                        }
                    };

                    template<typename T_TemperatureFunctor, typename T_ValueFunctor>
                    struct FreeTemperature
                        : public detail::TemperatureImpl<T_ValueFunctor>
                        , public particles::functor::User<T_TemperatureFunctor>
                    {
                        //! Base implementation class
                        using Base = detail::TemperatureImpl<T_ValueFunctor>;

                        //! Wrapper around user-provided functor
                        using UserFunctor = particles::functor::User<T_TemperatureFunctor>;

                        /** Create a functor instance, including instances for user functor and its wrapper
                         *
                         * @param currentStep current time iteration
                         */
                        FreeTemperature(uint32_t const currentStep, IdGenerator idGen)
                            : UserFunctor(currentStep, idGen)
                        {
                        }

                        /** Manipulate the momentum of the given macroparticle
                         *
                         * @tparam T_StandardNormalRng functor::misc::RngWrapper, standard
                         *                             normal random number generator type
                         * @tparam T_Particle particle type
                         *
                         * @param totalCellOffset total offset including all slides [in cells]
                         * @param standardNormalRng standard normal random number generator
                         * @param particle particle to be manipulated
                         */
                        template<typename T_StandardNormalRng, typename T_Particle>
                        HDINLINE void operator()(
                            DataSpace<simDim> const& totalCellOffset,
                            T_StandardNormalRng& standardNormalRng,
                            T_Particle& particle)
                        {
                            auto const unitLength = sim.unit.length();
                            auto const cellSize_SI = precisionCast<float_64>(sim.pic.getCellSize()) * unitLength;
                            auto const position_SI = (precisionCast<float_64>(totalCellOffset)
                                                      + precisionCast<float_64>(particle[position_]))
                                * cellSize_SI.shrink<simDim>();
                            auto const temperatureKeV = UserFunctor::operator()(position_SI, cellSize_SI);
                            Base::operator()(standardNormalRng, particle, temperatureKeV);
                        }
                    };

                } // namespace acc
            } // namespace unary
        } // namespace manipulators
    } // namespace particles
} // namespace picongpu
