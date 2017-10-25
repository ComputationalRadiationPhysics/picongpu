/**
 * \file
 * Copyright 2017 Alexander Matthes
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <boost/predef.h>
#if BOOST_COMP_INTEL == 0 // Work around for broken intel detection
    #if defined(__INTEL_COMPILER)
        #ifdef BOOST_COMP_INTEL_DETECTION
            #undef BOOST_COMP_INTEL_DETECTION
        #endif
        #define BOOST_COMP_INTEL_DETECTION BOOST_PREDEF_MAKE_10_VVRR(__INTEL_COMPILER)
        #if defined(BOOST_COMP_INTEL)
            #undef BOOST_COMP_INTEL
        #endif
        #define BOOST_COMP_INTEL BOOST_COMP_INTEL_DETECTION
    #endif
#endif
