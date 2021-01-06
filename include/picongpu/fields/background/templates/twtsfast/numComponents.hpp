/* Copyright 2014-2021 Alexander Debus, Axel Huebl
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
    namespace templates
    {
        namespace twtsfast
        {
            namespace detail
            {
                /** Number of field components used in the simulation. [Default: 3 for both 2D and 3D] */
                uint32_t const numComponents = 3;
            } /* namespace detail */
        } /* namespace twtsfast*/
    } /* namespace templates */
} /* namespace picongpu */
