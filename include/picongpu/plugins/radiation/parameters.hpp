/* Copyright 2013-2018 Axel Huebl, Rene Widera, Richard Pausch
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


#include "vector.hpp"
#include "picongpu/simulation_defines.hpp"


typedef cuda_vec<picongpu::float3_X, picongpu::float_X> vector_X;
typedef /*__align__(16)*/ cuda_vec<picongpu::float3_32, picongpu::float_32> vector_32;
typedef /*__align__(32)*/ cuda_vec<picongpu::float3_64, picongpu::float_64> vector_64;


