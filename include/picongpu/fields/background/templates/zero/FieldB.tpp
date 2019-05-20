/* Copyright 2019 Sergei Bastrakov
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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/fields/background/templates/zero/FieldB.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

#include <cstdint>


namespace picongpu
{
namespace fields
{
namespace background
{
namespace templates
{
namespace zero
{

    HDINLINE FieldB::FieldB( float3_64 const /* unitField */ )
    {
    }

    HDINLINE float3_X FieldB::operator( )(
        pmacc::DataSpace< simDim > const & /*cellIdx*/,
        uint32_t const /*currentStep*/
    ) const
    {
        return float3_X::create( 0._X );
    }

} // namespace zero
} // namespace templates
} // namespace background
} // namespace fields
} // namespace picongpu
