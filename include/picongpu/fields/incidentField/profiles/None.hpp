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

#include "picongpu/fields/incidentField/ZeroFunctor.hpp"
#include "picongpu/fields/incidentField/profiles/None.def"
#include "picongpu/fields/incidentField/traits/GetAmplitude.hpp"
#include "picongpu/fields/incidentField/traits/GetFunctor.hpp"

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace profiles
            {
                struct None
                {
                    //! Get text name of the incident field profile
                    HINLINE static std::string getName()
                    {
                        return "None";
                    }

                    static nlohmann::json metadata()
                    {
                        return nlohmann::json::object();
                    }
                };
            } // namespace profiles
            namespace traits
            {
                namespace detail
                {
                    //! Get type of incident field E functor for the none profile type
                    template<>
                    struct GetFunctorIncidentE<profiles::None>
                    {
                        using type = ZeroFunctor;
                    };

                    //! Get type of incident field B functor for the none profile type
                    template<>
                    struct GetFunctorIncidentB<profiles::None>
                    {
                        using type = ZeroFunctor;
                    };

                    //! None profile has no phase velocity, use c as a placeholder value
                    template<>
                    struct GetPhaseVelocity<profiles::None>
                    {
                        HINLINE float_X operator()() const
                        {
                            return sim.pic.getSpeedOfLight();
                        }
                    };

                } // namespace detail
                //! Specialization for None profile which has no amplitude
                template<>
                struct GetAmplitude<profiles::None>
                {
                    static constexpr float_X value = 0.0_X;
                };
            } // namespace traits
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
