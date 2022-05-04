/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#    if _OPENMP < 201307
#        error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#    endif

#    include <alpaka/core/Common.hpp>

#    include <omp.h>

#    include <cstddef>
#    include <iostream>
#    include <sstream>
#    include <stdexcept>
#    include <string>
#    include <type_traits>
#    include <utility>

namespace alpaka::omp5::detail
{
    //! OMP5 runtime API error checking with log and exception, ignoring specific error values
    ALPAKA_FN_HOST inline auto omp5Check(int const& error, char const* desc, char const* file, int const& line) -> void
    {
        if(error != 0)
        {
            std::ostringstream os;
            os << std::string(file) << "(" << std::to_string(line) << ") " << std::string(desc) << " : '" << error
               << "': '"
               << "TODO"
               << "'!";
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
            std::cerr << os.str() << std::endl;
#    endif
            ALPAKA_DEBUG_BREAK;
            throw std::runtime_error(os.str());
        }
    }
} // namespace alpaka::omp5::detail

//! OMP5 runtime error checking with log and exception.
#    define ALPAKA_OMP5_CHECK(cmd) ::alpaka::omp5::detail::omp5Check(cmd, #    cmd, __FILE__, __LINE__)

#endif
