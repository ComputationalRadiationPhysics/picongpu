/* Copyright 2022-2024 Sergei Bastrakov, Julian Lenz
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

#include "picongpu/defines.hpp"
#include "picongpu/fields/incidentField/profiles/Free.def"
#include "picongpu/fields/incidentField/traits/GetAmplitude.hpp"
#include "picongpu/fields/incidentField/traits/GetFunctor.hpp"
#include "picongpu/traits/GetMetadata.hpp"

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

                    static nlohmann::json metadata()
                    {
                        auto result = nlohmann::json::object();
                        result["name"] = "Free";
                        result["B field"] = functorMetadata<T_FunctorIncidentB>();
                        result["E field"] = functorMetadata<T_FunctorIncidentE>();
                        return result;
                    }

                private:
                    template<typename Functor, std::enable_if_t<providesMetadataAtCT<Functor>, bool> = true>
                    static auto functorMetadata()
                    {
                        return Functor::metadata();
                    }

                    template<typename Functor, std::enable_if_t<!providesMetadataAtCT<Functor>, bool> = true>
                    static nlohmann::json functorMetadata()
                    {
                        return "no metadata available";
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
