/* Copyright 2013-2021 Axel Huebl, Rene Widera
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

#include <stdint.h>
#include <pmacc/debug/VerboseLog.hpp>

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
