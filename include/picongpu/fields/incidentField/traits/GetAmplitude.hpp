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
