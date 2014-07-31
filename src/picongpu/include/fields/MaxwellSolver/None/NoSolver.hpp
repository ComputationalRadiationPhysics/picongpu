/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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

#include "NoSolver.def"

#include "types.h"
#include "simulation_defines.hpp"


namespace picongpu
{
    namespace noSolver
    {
        using namespace PMacc;


        class NoSolver
        {
        private:
            typedef MappingDesc::SuperCellSize SuperCellSize;

            MappingDesc cellDescription;

            template<uint32_t AREA>
            void updateE()
            {
                return;
            }

            template<uint32_t AREA>
            void updateBHalf()
            {
                return;
            }

        public:

            NoSolver(MappingDesc cellDescription) : cellDescription(cellDescription)
            {

            }

            void update_beforeCurrent(uint32_t)
            {

            }

            void update_afterCurrent(uint32_t)
            {

            }
        };

    } // namespace noSolver

} // picongpu
