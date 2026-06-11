/*
 * SPDX-FileCopyrightText: Sergei Bastrakov, Julian Lenz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/fields/incidentField/ZeroFunctor.hpp"
#include "picongpu/fields/incidentField/profiles/None.def"
#include "picongpu/fields/incidentField/traits/GetAmplitude.hpp"
#include "picongpu/fields/incidentField/traits/GetFunctor.hpp"
#include "picongpu/fields/incidentField/traits/GetPhaseVelocity.hpp"

#include <cstdint>
#include <string>

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
