/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/types.hpp>

#include <picongpu/logging.hpp>

namespace picongpu
{
    using namespace pmacc;
} // namespace picongpu

// clang-format off
#include "picongpu/param/precision.param"
#include "picongpu/param/dimension.param"
#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
#    include "picongpu/param/mallocMC.param"
#endif
#include "picongpu/param/memory.param"
#include "picongpu/param/random.param"
#include "picongpu/param/physicalConstants.param"
#include "picongpu/param/speciesConstants.param"
#include "picongpu/param/simulation.param"
#include "picongpu/unitless/simulation.unitless"
#include "picongpu/param/speciesAttributes.param"
// clang-format on
