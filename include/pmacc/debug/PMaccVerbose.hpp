/* Copyright 2013-2021 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/debug/VerboseLog.hpp"

#include <stdint.h>

#ifndef PMACC_VERBOSE_LVL
#    define PMACC_VERBOSE_LVL 0
#endif

namespace pmacc
{
    /*create verbose class*/
    DEFINE_VERBOSE_CLASS(PMaccVerbose)
    (
        /* define log lvl for later use
         * e.g. log<pmaccLogLvl::NOTHING>("TEXT");*/
        DEFINE_LOGLVL(0, NOTHING); DEFINE_LOGLVL(1, MEMORY); DEFINE_LOGLVL(2, INFO); DEFINE_LOGLVL(4, CRITICAL);
        DEFINE_LOGLVL(8, MPI);
        DEFINE_LOGLVL(16, CUDA_RT);
        DEFINE_LOGLVL(32, COMMUNICATION);
        DEFINE_LOGLVL(64, EVENT);)
        /*set default verbose lvl (integer number)*/
        (NOTHING::lvl | PMACC_VERBOSE_LVL);

    // short name for access verbose types of PMacc
    using ggLog = PMaccVerbose;

} // namespace pmacc
