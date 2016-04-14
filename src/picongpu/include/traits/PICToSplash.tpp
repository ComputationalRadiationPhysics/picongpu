/**
 * Copyright 2013-2016 Axel Huebl
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
#include <splash/splash.h>

#include "simulation_defines.hpp"
#include <boost/mpl/if.hpp>
#include <boost/type_traits.hpp>

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
    /** Trait for float_32 */
    template<>
    struct PICToSplash<float_32>
    {
        typedef splash::ColTypeFloat type;
    };

    /** Trait for float_64 */
    template<>
    struct PICToSplash<float_64>
    {
        typedef splash::ColTypeDouble type;
    };

    /** Trait for int */
    template<>
    struct PICToSplash<int>
    {
        typedef splash::ColTypeInt type;
    };

    template<>
    struct PICToSplash<int64_t>
    {
        typedef splash::ColTypeInt64 type;
    };

    template<>
    struct PICToSplash<uint64_t>
    {
        typedef splash::ColTypeUInt64 type;
    };

    /** Specialization for uint64_cu.
     *  If uint64_cu happens to be the same as uint64_t we use an unused dummy type
     *  to avoid duplicate specialization
     */
    struct uint64_cu_unused_splash;
    template<>
    struct PICToSplash<
                        typename bmpl::if_<
                            boost::is_same<uint64_t, uint64_cu>,
                            uint64_cu_unused_splash,
                            uint64_cu
                        >::type
                     >: public PICToSplash<uint64_t>
    {};

    /** Trait for splash::Dimensions */
    template<>
    struct PICToSplash<splash::Dimensions>
    {
        typedef splash::ColTypeDim type;
    };

} //namespace traits

}// namespace picongpu

#endif // (ENABLE_HDF5==1)
