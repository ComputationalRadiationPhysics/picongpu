/* Copyright 2013-2021 Axel Huebl, Rene Widera, Richard Pausch
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
#include "picongpu/plugins/radiation/vector.hpp"


namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            using vector_X = cuda_vec<picongpu::float3_X, picongpu::float_X>;
            using vector_32 = /*__align__(16)*/ cuda_vec<picongpu::float3_32, picongpu::float_32>;
            using vector_64 = /*__align__(32)*/ cuda_vec<picongpu::float3_64, picongpu::float_64>;
        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
