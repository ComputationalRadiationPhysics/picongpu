/*
 * SPDX-FileCopyrightText: Sergei Bastrakov, Julian Lenz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/incidentField/profiles/Free.def"
#include "picongpu/fields/incidentField/traits/GetAmplitude.hpp"
#include "picongpu/fields/incidentField/traits/GetFunctor.hpp"
#include "picongpu/fields/incidentField/traits/GetPhaseVelocity.hpp"

#include <cstdint>
#include <string>
#include <type_traits>

#include <nlohmann/json.hpp>

namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace profiles
            {
                template<typename T_FunctorIncidentE, typename T_FunctorIncidentB>
                struct Free
                {
                    //! Get text name of the incident field profile
                    HINLINE static std::string getName()
                    {
                        return "Free";
                    }
                };
            } // namespace profiles

            namespace traits
            {
                namespace detail
                {
                    /** Get type of incident field E functor for the free profile type
                     *
                     * @tparam T_FunctorIncidentE functor for the incident E field
                     * @tparam T_FunctorIncidentB functor for the incident B field
                     */
                    template<typename T_FunctorIncidentE, typename T_FunctorIncidentB>
                    struct GetFunctorIncidentE<profiles::Free<T_FunctorIncidentE, T_FunctorIncidentB>>
                    {
                        using type = T_FunctorIncidentE;
                    };

                    /** Get type of incident field B functor for the free profile type
                     *
                     * @tparam T_FunctorIncidentE functor for the incident E field
                     * @tparam T_FunctorIncidentB functor for the incident B field
                     */
                    template<typename T_FunctorIncidentE, typename T_FunctorIncidentB>
                    struct GetFunctorIncidentB<profiles::Free<T_FunctorIncidentE, T_FunctorIncidentB>>
                    {
                        using type = T_FunctorIncidentB;
                    };

                    //! Free profile has an unknown phase velocity, use c as a default value
                    template<typename T_FunctorIncidentE, typename T_FunctorIncidentB>
                    struct GetPhaseVelocity<profiles::Free<T_FunctorIncidentE, T_FunctorIncidentB>>
                    {
                        HINLINE float_X operator()() const
                        {
                            return sim.pic.getSpeedOfLight();
                        }
                    };

                } // namespace detail

                //! Specialization for Free profile which has unknown amplitude
                template<typename T_FunctorIncidentE, typename T_FunctorIncidentB>
                struct GetAmplitude<profiles::Free<T_FunctorIncidentE, T_FunctorIncidentB>>
                {
                    static constexpr float_X value = 0.0_X;
                };
            } // namespace traits
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
