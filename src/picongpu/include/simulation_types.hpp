/**
 * Copyright 2013-2015 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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
#include "traits/GetComponentsType.hpp"
#include "traits/NumberOfExchanges.hpp"
#include "traits/GetDataBoxType.hpp"

namespace picongpu
{

//! define all elements which can send and resive

enum CommunicationTag
{
    NO_COMMUNICATION = 0u,
    FIELD_B = 1u,
    FIELD_E = 2u,
    FIELD_J = 3u,
    FIELD_JRECV = 4u,
    FIELD_TMP = 5u,
    SPECIES_FIRSTTAG = 42u
};


//! defines field types some various methods (e.g. Laser::manipulate)

enum FieldType
{
    FIELD_TYPE_E, FIELD_TYPE_B, FIELD_TYPE_TMP
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
