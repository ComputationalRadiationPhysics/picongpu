/* Copyright 2013-2021 Axel Huebl
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

#if(ENABLE_HDF5 == 1)
#    include <splash/splash.h>

#    include "picongpu/simulation_defines.hpp"

namespace picongpu
{
    namespace traits
    {
        template<>
        struct SplashToPIC<splash::ColTypeBool>
        {
            typedef bool type;
        };

        template<>
        struct SplashToPIC<splash::ColTypeFloat>
        {
            typedef float_32 type;
        };

        template<>
        struct SplashToPIC<splash::ColTypeDouble>
        {
            typedef float_64 type;
        };

        /** Native int */
        template<>
        struct SplashToPIC<splash::ColTypeInt>
        {
            typedef int type;
        };

        template<>
        struct SplashToPIC<splash::ColTypeInt32>
        {
            typedef int32_t type;
        };

        template<>
        struct SplashToPIC<splash::ColTypeUInt32>
        {
            typedef uint32_t type;
        };

        template<>
        struct SplashToPIC<splash::ColTypeInt64>
        {
            typedef int64_t type;
        };

        template<>
        struct SplashToPIC<splash::ColTypeUInt64>
        {
            typedef uint64_t type;
        };

        template<>
        struct SplashToPIC<splash::ColTypeDim>
        {
            typedef splash::Dimensions type;
        };

    } // namespace traits

} // namespace picongpu

#endif // (ENABLE_HDF5==1)
