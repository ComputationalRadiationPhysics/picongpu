/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/debug/VerboseLogMakros.hpp"

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
