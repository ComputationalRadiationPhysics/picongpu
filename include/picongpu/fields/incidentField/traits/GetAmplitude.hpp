/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/incidentField/traits/GetFunctor.hpp"

namespace picongpu::fields::incidentField::traits
{
    /** Get max E field amplitude for the given profile type
     *
     * The resulting value is set as ::value, in internal units.
     * This trait has to be specialized by all profiles.
     *
     * @tparam T_Profile profile type
     */

    template<typename T_Profile>
    struct GetAmplitude
    {
        using FunctorE = detail::FunctorIncidentE<T_Profile>;
        static constexpr float_X value = FunctorE::Unitless::AMPLITUDE;
    };

    /** Max E field amplitude in internal units for the given profile type
     *
     * @tparam T_Profile profile type
     */
    template<typename T_Profile>
    constexpr float_X amplitude = GetAmplitude<T_Profile>::value;
} // namespace picongpu::fields::incidentField::traits
