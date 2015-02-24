/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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

#include "version.hpp"
#include "algorithms/TypeCast.hpp"
#include "algorithms/PromoteType.hpp"
#include "algorithms/ForEach.hpp"
#include "algorithms/math.hpp"
#include "traits/GetMargin.hpp"
#include "traits/SplashToPIC.hpp"
#include "traits/PICToSplash.hpp"
#include "traits/GetComponentsType.hpp"
#include "traits/NumberOfExchanges.hpp"

namespace picongpu
{

//! defines form of particle

enum ParticleType
{
    ION = 0, ELECTRON = 1
};

//! define all elements which can send and resive

enum CommunicationTag
{
    FIELD_B = 0u, FIELD_E = 1u, FIELD_J = 2u, FIELD_TMP = 3u,
    PAR_IONS = 4u, PAR_ELECTRONS = 5u,
    NO_COMMUNICATION = 16u
};


//! define the place where data is stored

enum DataPlace
{
    DEVICE, HOST
};

//! defines field types some various methods (e.g. Laser::manipulate)

enum FieldType
{
    FIELD_TYPE_E, FIELD_TYPE_B, FIELD_TYPE_TMP
};

enum Seeds
{
    TEMPERATURE_SEED = 255845, POSITION_SEED = 854666252
};

namespace precision32Bit
{
typedef float precisionType;
}

namespace precision64Bit
{
typedef double precisionType;
}

namespace math = PMacc::algorithms::math;
using namespace PMacc::algorithms::precisionCast;
using namespace PMacc::algorithms::promoteType;
using namespace PMacc::algorithms::forEach;
using namespace PMacc::traits;
using namespace picongpu::traits;

}




