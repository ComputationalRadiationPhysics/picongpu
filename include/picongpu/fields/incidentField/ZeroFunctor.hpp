/*
 * SPDX-FileCopyrightText: Sergei Bastrakov, Finn-Ole Carstens
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu::fields::incidentField
{
    //! Helper incident field functor always returning 0
    struct ZeroFunctor
    {
        /** Create a functor on the host side for the given time step
         *
         * @param currentStep current time step index, note that it is fractional
         * @param unitField conversion factor from SI to internal units,
         *                  field_internal = field_SI / unitField
         */
        HINLINE ZeroFunctor(float_X const currentStep, float3_64 const unitField)
        {
        }

        /** Return zero incident field for any given position
         *
         * @param totalCellIdx cell index in the total domain (including all moving window slides),
         *        note that it is fractional
         * @return incident field value in internal units
         */
        HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
        {
            return float3_X::create(0.0_X);
        }
    };
} // namespace picongpu::fields::incidentField
