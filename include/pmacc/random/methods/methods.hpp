/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/random/methods/AlpakaRand.hpp"
#ifndef ALPAKA_DISABLE_VENDOR_RNG
#    include "pmacc/random/methods/MRG32k3aMin.hpp"
#    include "pmacc/random/methods/XorMin.hpp"
#endif
