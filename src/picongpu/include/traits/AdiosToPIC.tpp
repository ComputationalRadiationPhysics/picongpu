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
    /** Trait for adios_integer */
    template<>
    struct AdiosToPIC<adios_integer>
    {
        typedef int type;
    };
    
    /** Trait for adios_unsigned_integer */
    template<>
    struct AdiosToPIC<adios_unsigned_integer>
    {
        typedef unsigned int type;
    };

    template<>
    struct AdiosToPIC<adios_unsigned_long>
    {
        typedef uint64_t type;
    };

    /** Trait for adios_real */
    template<>
    struct AdiosToPIC<adios_real>
    {
        typedef float_32 type;
    };
    
    /** Trait for adios_double */
    template<>
    struct AdiosToPIC<adios_double>
    {
        typedef float_64 type;
    };

} //namespace traits

}// namespace picongpu

#endif // (ENABLE_ADIOS==1)
