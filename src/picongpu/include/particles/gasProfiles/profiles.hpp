/**
 * Copyright 2014-2016 Rene Widera
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

#include "particles/gasProfiles/IProfile.hpp"
#include "particles/gasProfiles/FreeFormulaImpl.hpp"
#include "particles/gasProfiles/GaussianImpl.hpp"
#include "particles/gasProfiles/HomogenousImpl.hpp"
#include "particles/gasProfiles/LinearExponentialImpl.hpp"
#include "particles/gasProfiles/GaussianCloudImpl.hpp"
#include "particles/gasProfiles/SphereFlanksImpl.hpp"

#if (ENABLE_HDF5 == 1)
#include "particles/gasProfiles/FromHDF5Impl.hpp"
#endif
