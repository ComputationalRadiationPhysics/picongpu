/* Copyright 2013-2021 Rene Widera, Richard Pausch
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

#include <iostream>
#include "VectorTypes.hpp"

namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            HINLINE void check_consistency(void)
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
