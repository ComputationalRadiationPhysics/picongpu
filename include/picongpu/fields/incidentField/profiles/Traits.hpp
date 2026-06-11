/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

namespace picongpu::fields::incidentField::profiles
{
    /** Express if the Huygens surface in transversal directions must be extended to the simulation borders in case the
     * transversal direction is periodic.
     *
     * The result of this trait should be true in case you have non-zero field amplitudes on the transversal borders
     * else the field will not be contiguous if periodic boundaries are enabled.
     *
     * @tparam T_Profile incident field profile
     */
    template<typename T_Profile>
    struct MakePeriodicTransversalHuygensSurfaceContiguous
    {
        static constexpr bool value = false;
    };

    /** shorthand notation for @see MakePeriodicTransversalHuygensSurfaceContiguous */
    template<typename T_Profile>
    constexpr bool makePeriodicTransversalHuygensSurfaceContiguous
        = MakePeriodicTransversalHuygensSurfaceContiguous<T_Profile>::value;

} // namespace picongpu::fields::incidentField::profiles
