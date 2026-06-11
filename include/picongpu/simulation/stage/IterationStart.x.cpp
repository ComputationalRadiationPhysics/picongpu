/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#include "picongpu/simulation/stage/IterationStart.hpp"

#include "picongpu/defines.hpp"
#include "picongpu/param/iterationStart.param"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/param.hpp"

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
