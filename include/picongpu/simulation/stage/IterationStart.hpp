/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/functor/Call.hpp>

#include <cstdint>

namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** Functor for the very first stage of the PIC loop
             *
             * Calls functors defined in iterationStart.param
             */
            struct IterationStart
            {
                /** Call all iteration start functors
                 *
                 * @param step index of time iteration
                 */
                void operator()(uint32_t const step) const;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
