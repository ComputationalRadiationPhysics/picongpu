/* Copyright 2024 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software you can redistribute it and or modify
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

//! @file defines the interfaces for the modules of the SpawnFromSourceSpecies kernel framework

#pragma once

#include "picongpu/particles/creation/moduleInterfaces/AdditionalDataIndexFunctor.hpp"
#include "picongpu/particles/creation/moduleInterfaces/InitKernelStateFunctor.hpp"
#include "picongpu/particles/creation/moduleInterfaces/ParticlePairUpdateFunctor.hpp"
#include "picongpu/particles/creation/moduleInterfaces/PredictorFunctor.hpp"
#include "picongpu/particles/creation/moduleInterfaces/SanityCheckInputs.hpp"
#include "picongpu/particles/creation/moduleInterfaces/SuperCellFilterFunctor.hpp"
#include "picongpu/particles/creation/moduleInterfaces/WriteOutKernelStateFunctor.hpp"
