/**
 * Copyright 2013-2016 Axel Huebl, Felix Schmitt
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

#if (ENABLE_ADIOS==1)
#include <adios.h>

#include "simulation_defines.hpp"

namespace picongpu
{

namespace traits
{
    /** Trait for int */
    template<>
    struct PICToAdios<int>
    {
        ADIOS_DATATYPES type;
        
        PICToAdios() :
        type(adios_integer) {}
    };
    
    /** Trait for uint32_t */
    template<>
    struct PICToAdios<uint32_t>
    {
        ADIOS_DATATYPES type;
        
        PICToAdios() :
        type(adios_unsigned_integer) {}
    };
    
    /** Trait for uint64_t */
    template<>
    struct PICToAdios<uint64_t>
    {
        ADIOS_DATATYPES type;
        
        PICToAdios() :
        type(adios_unsigned_long) {}
    };

    template<>
    struct PICToAdios<uint64_cu>
    {
        ADIOS_DATATYPES type;

        PICToAdios() :
        type(adios_unsigned_long) {}
    };

    /** Trait for float_32 */
    template<>
    struct PICToAdios<float_32>
    {
        ADIOS_DATATYPES type;
        
        PICToAdios() :
        type(adios_real) {}
    };

    /** Trait for float_64 */
    template<>
    struct PICToAdios<float_64>
    {
        ADIOS_DATATYPES type;
        
        PICToAdios() :
        type(adios_double) {}
    };

} //namespace traits

}// namespace picongpu

#endif // (ENABLE_ADIOS==1)
