/* Copyright 2023 Rene Widera
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

namespace picongpu::fields::incidentField::profiles
{
    /** Express if the Huyhens surface in transversal directions must be extended to the simulation borders in case the
     * transversal direction is periodic.
     *
     * The result of this trait should be true in case you have non zero field amplitudes on the transversal borders
     * else the field will not be contiguous if periodic boundaries are enabled.
     *
     * @tparam T_Profile incident field profile
     */
    template<typename T_Profile>
    struct MakePeriodicTransversalHuygensSurfaceContiguous
    {
        static constexpr bool value = false;
    };


    /** short hand notation for @see MakePeriodicTransversalHuygensSurfaceContiguous */
    template<typename T_Profile>
    constexpr bool makePeriodicTransversalHuygensSurfaceContiguous
        = MakePeriodicTransversalHuygensSurfaceContiguous<T_Profile>::value;

} // namespace picongpu::fields::incidentField::profiles
