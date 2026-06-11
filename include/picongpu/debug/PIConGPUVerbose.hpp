/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/debug/VerboseLogMakros.hpp>

namespace picongpu
{
#ifndef PIC_VERBOSE_LVL
#    define PIC_VERBOSE_LVL 0
#endif

    /*create verbose class*/
    DEFINE_VERBOSE_CLASS(PIConGPUVerbose)
    (
        /* define log lvl for later use
         * e.g. log<pmaccLogLvl::NOTHING>("TEXT");*/
        DEFINE_LOGLVL(0, NOTHING); DEFINE_LOGLVL(1, PHYSICS); DEFINE_LOGLVL(2, DOMAINS); DEFINE_LOGLVL(4, CRITICAL);
        DEFINE_LOGLVL(8, MEMORY);
        DEFINE_LOGLVL(16, SIMULATION_STATE);
        DEFINE_LOGLVL(32, INPUT_OUTPUT);)
        /*set default verbose lvl (integer number)*/
        (NOTHING::lvl | PIC_VERBOSE_LVL);


} /* namespace picongpu */
