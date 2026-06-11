/*
 * SPDX-FileCopyrightText: Rene Widera, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "picongpu/plugins/radiation/param.hpp"

#include <pmacc/debug/VerboseLogMakros.hpp>

namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            /*create verbose class*/
            DEFINE_VERBOSE_CLASS(PIConGPUVerboseRadiation)
            (
                /* define log levels for later use
                 * e.g. log<pmaccLogLvl::NOTHING>("TEXT");*/
                DEFINE_LOGLVL(0, NOTHING); DEFINE_LOGLVL(1, PHYSICS); DEFINE_LOGLVL(2, SIMULATION_STATE);
                DEFINE_LOGLVL(4, MEMORY);
                DEFINE_LOGLVL(8, CRITICAL);)
                /*set default verbose levels (integer number)*/
                (NOTHING::lvl | PIC_VERBOSE_RADIATION);

        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
