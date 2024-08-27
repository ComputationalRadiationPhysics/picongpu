/* Copyright 2021-2023 Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#include "picongpu/simulation/stage/IterationStart.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/param/iterationStart.param"
#include "picongpu/particles/filter/filter.hpp"

#include <pmacc/functor/Call.hpp>

#include <cstdint>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            void IterationStart::operator()(uint32_t const step) const
            {
                meta::ForEach<IterationStartPipeline, pmacc::functor::Call<boost::mpl::_1>> callFunctors;
                callFunctors(step);
            }

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
