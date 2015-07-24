/**
 * Copyright 2015 Rene Widera
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

#include "simulation_defines.hpp"
#include "traits/GetFlagType.hpp"
#include "traits/Resolve.hpp"

#include <boost/mpl/if.hpp>

namespace picongpu
{
namespace traits
{

namespace detail
{
    value_identifier(float_X, DefaultDensityRatio, 1.0);
} //namespace detail


/** get density ratio of a species
 *
 * ratio is set to 1.0 if no alias `densityRatio<>` is defined
 *
 * @treturn ::type `value_identifier` with the default density
 */
template<typename T_Species>
struct GetDensityRatio
{
    typedef typename T_Species::FrameType FrameType;
    typedef typename HasFlag<FrameType, densityRatio<> >::type hasDensityRatio;
    typedef typename PMacc::traits::Resolve<
        typename GetFlagType<
            FrameType, densityRatio<>
        >::type
    >::type DensityRatioOfSpecies;

    typedef typename bmpl::if_<
         hasDensityRatio,
        DensityRatioOfSpecies,
        detail::DefaultDensityRatio
    >::type type;
};

} //namespace traits

}// namespace picongpu
