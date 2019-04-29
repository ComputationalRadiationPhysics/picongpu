/* Copyright 2013-2019 Axel Huebl, Rene Widera
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

#include "picongpu/fields/MaxwellSolver/None/None.hpp"
#include "picongpu/fields/MaxwellSolver/Yee/Yee.hpp"
#if (SIMDIM==3)
#include "picongpu/fields/MaxwellSolver/Lehe/Lehe.hpp"
#if( PMACC_CUDA_ENABLED == 1 )
#   include "picongpu/fields/MaxwellSolver/DirSplitting/DirSplitting.hpp"
#endif
#endif
