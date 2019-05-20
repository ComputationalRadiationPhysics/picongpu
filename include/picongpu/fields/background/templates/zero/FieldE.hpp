/* Copyright 2014-2019 Axel Huebl, Alexander Debus, Sergei Bastrakov
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

    //! Zero E field background
    class FieldE
    {
    public:

        //! Does not contribute to particle pushing
        static constexpr bool InfluenceParticlePusher = false;

        /** Construct the background field
         *
         * @param unitField PIC unit for the field
         */
        HDINLINE FieldE( float3_64 const unitField );

        /** Background values are always zero
         *
         * \param cellIdx total cell index counted from the start at t = 0
         * \param currentStep current time iteration
         */
        HDINLINE float3_X
        operator( )(
            pmacc::DataSpace< simDim > const & cellIdx,
            uint32_t const currentStep
        ) const;

    };

} // namespace zero
} // namespace templates
} // namespace background
} // namespace fields
} // namespace picongpu
