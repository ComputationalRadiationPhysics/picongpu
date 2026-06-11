/*
 * SPDX-FileCopyrightText: Rene Widera, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "VectorTypes.hpp"
#include "picongpu/plugins/radiation/param.hpp"

#include <iostream>

namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            HINLINE void checkConsistency(void)
            {
                using namespace parameters;
                std::cout << " checking efficiency of radiation code: ";
                if(radiation_frequencies::N_omega % radiation_frequencies::blocksize_omega == 0)
                    std::cout << "OK" << std::endl;
                else
                    std::cout << "better use power of two for N_omega" << std::endl;
                // \@todo is there a way to do this with  compile time asserts???
            }

        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
