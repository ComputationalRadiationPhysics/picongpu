/**
 * Copyright 2013 Axel Huebl
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

#if (ENABLE_HDF5==1)
#include <splash.h>

#include "simulation_defines.hpp"

namespace picongpu
{

namespace traits
{
    /** Trait for bool */
    template<>
    struct PICToSplash<bool>
    {
        typedef splash::ColTypeBool type;
    };
    /** Trait for float */
    template<>
    struct PICToSplash<float>
    {
        typedef splash::ColTypeFloat type;
    };

    /** Trait for double */
    template<>
    struct PICToSplash<double>
    {
        typedef splash::ColTypeDouble type;
    };

    /** Trait for int */
    template<>
    struct PICToSplash<int>
    {
        typedef splash::ColTypeInt type;
    };

    /** Trait for splash::Dimensions */
    template<>
    struct PICToSplash<splash::Dimensions>
    {
        typedef splash::ColTypeDim type;
    };

} //namespace traits

}// namespace picongpu

#endif // (ENABLE_HDF5==1)
