/**
 * Copyright 2014-2015 Alexander Debus
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

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_classTypes.hpp"

#include "math/Vector.hpp"
#include "dimensions/DataSpace.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "math/Complex.hpp"
#include "fields/background/templates/TWTS/TWTS.hpp"

#include "fields/background/templates/TWTS/RotateField.tpp"
#include "fields/background/templates/TWTS/Get_tdelay_SI.tpp"
#include "fields/background/templates/TWTS/GetFieldPositions_SI.tpp"
#include "fields/background/templates/TWTS/TWTSFieldE.tpp"
#include "fields/background/templates/TWTS/TWTSFieldB.tpp"
