/* Copyright 2020-2023 Sergei Bastrakov
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

#include "picongpu/fields/MaxwellSolver/DispersionRelationSolver.hpp"

#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/meta/conversion/Unique.hpp>

#include <cstdint>
#include <type_traits>


namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace detail
            {
                /** Get type of incident field functor for the given profile type
                 *
                 * The resulting functor is set as ::type.
                 * These traits have to be specialized by all profiles.
                 *
                 * @tparam T_Profile profile type
                 *
                 * @{
                 */

                //! Get functor for incident E values
                template<typename T_Profile>
                struct GetFunctorIncidentE;

                //! Get functor for incident B values
                template<typename T_Profile>
                struct GetFunctorIncidentB;

                /** @} */

                /** Type of incident E/B functor for the given profile type
                 *
                 * These are helper aliases to wrap GetFunctorIncidentE/B.
                 * The latter present customization points.
                 *
                 * @tparam T_Profile profile type
                 *
                 * @{
                 */

                //! Functor for incident E values
                template<typename T_Profile>
                using FunctorIncidentE = typename GetFunctorIncidentE<T_Profile>::type;

                //! Functor for incident B values
                template<typename T_Profile>
                using FunctorIncidentB = typename GetFunctorIncidentB<T_Profile>::type;

                /** @} */

                /** Calculate phase velocity for the enabled field solver and given unitless parameters
                 *
                 * @tparam T_Unitless unitless parameters type, must be compatible to
                 * profiles::detail::BaseParamUnitless
                 */
                template<typename T_Unitless>
                HINLINE float_X calculatePhaseVelocity()
                {
                    auto const omega = pmacc::math::Pi<float_64>::doubleValue
                        * static_cast<float_64>(SPEED_OF_LIGHT / T_Unitless::WAVE_LENGTH);
                    // Assume propagation along y as all laser profiles do it
                    auto const direction = float3_64{T_Unitless::DIR_X, T_Unitless::DIR_Y, T_Unitless::DIR_Z};
                    auto const absK = maxwellSolver::DispersionRelationSolver<Solver>{}(omega, direction);
                    auto const phaseVelocity = omega / absK / SPEED_OF_LIGHT;
                    return static_cast<float_X>(phaseVelocity);
                }

                /** Get phase velocity for the enabled field solver and given incident field profile
                 *
                 * @tparam T_Profile profile type
                 *
                 * @{
                 */

                //! General implementation for parametrized profiles with parameters compatible to profiles::BaseParam
                template<typename T_Profile>
                struct GetPhaseVelocity
                {
                    HINLINE float_X operator()() const
                    {
                        using Functor = FunctorIncidentE<T_Profile>;
                        using Unitless = typename Functor::Unitless;
                        return calculatePhaseVelocity<Unitless>();
                    }
                };

                //! None profile has no phase velocity, use c as a placeholder value
                template<>
                struct GetPhaseVelocity<profiles::None>
                {
                    HINLINE float_X operator()() const
                    {
                        return SPEED_OF_LIGHT;
                    }
                };

                //! Free profile has an unknown phase velocity, use c as a default value
                template<typename T_FunctorIncidentE, typename T_FunctorIncidentB>
                struct GetPhaseVelocity<profiles::Free<T_FunctorIncidentE, T_FunctorIncidentB>>
                {
                    HINLINE float_X operator()() const
                    {
                        return SPEED_OF_LIGHT;
                    }
                };

#if(ENABLE_OPENPMD == 1) && (SIMDIM == DIM3)
                //! InsightPulse profile has an unknown phase velocity, use c as a default value
                template<typename T_Params>
                struct GetPhaseVelocity<profiles::InsightPulse<T_Params>>
                {
                    HINLINE float_X operator()() const
                    {
                        return SPEED_OF_LIGHT;
                    }
                };
#endif

                /** @} */

            } // namespace detail

            /** Get max E field amplitude for the given profile type
             *
             * The resulting value is set as ::value, in internal units.
             * This trait has to be specialized by all profiles.
             *
             * @tparam T_Profile profile type
             *
             * @{
             */

            //! Generic implementation for all profiles with parameter structs
            template<typename T_Profile>
            struct GetAmplitude
            {
                using FunctorE = detail::FunctorIncidentE<T_Profile>;
                static constexpr float_X value = FunctorE::Unitless::AMPLITUDE;
            };

            //! Specialization for None profile which has no amplitude
            template<>
            struct GetAmplitude<profiles::None>
            {
                static constexpr float_X value = 0.0_X;
            };

            //! Specialization for Free profile which has unknown amplitude
            template<typename T_FunctorIncidentE, typename T_FunctorIncidentB>
            struct GetAmplitude<profiles::Free<T_FunctorIncidentE, T_FunctorIncidentB>>
            {
                static constexpr float_X value = 0.0_X;
            };

#if(ENABLE_OPENPMD == 1) && (SIMDIM == DIM3)
            //! Specialization for InsightPulse profile which has unknown amplitude
            // kann man die Templates so leer lassen und es fkt trd?
            template<typename T_Params>
            struct GetAmplitude<profiles::InsightPulse<T_Params>>
            {
                static constexpr float_X value = 0.0_X;
            };
#endif

            /** @} */

            /** Max E field amplitude in internal units for the given profile type
             *
             * @tparam T_Profile profile type
             */
            template<typename T_Profile>
            constexpr float_X amplitude = GetAmplitude<T_Profile>::value;

            //! Typelist of all enabled profiles, can contain duplicates
            using EnabledProfiles = pmacc::MakeSeq_t<
                XMin,
                XMax,
                YMin,
                YMax,
                std::conditional_t<simDim == 3, pmacc::MakeSeq_t<ZMin, ZMax>, pmacc::MakeSeq_t<>>>;

            //! Typelist of all unique enabled profiles, can contain duplicates
            using UniqueEnabledProfiles = pmacc::Unique_t<EnabledProfiles>;

            /** Get phase velocity for the enabled field solver and given incident field profile
             *
             * @tparam T_Profile profile type
             */
            template<typename T_Profile>
            HINLINE float_X getPhaseVelocity()
            {
                return detail::GetPhaseVelocity<T_Profile>{}();
            }

        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
