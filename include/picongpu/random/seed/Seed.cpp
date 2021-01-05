/* Copyright 2018-2021 Rene Widera
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

#include "picongpu/random/seed/Seed.hpp"

#include <mpi.h>
#include <string>
#include <chrono>
#include <cstdlib>


namespace picongpu
{
    namespace random
    {
        namespace seed
        {
            uint32_t FromTime::operator()() const
            {
                auto now = std::chrono::system_clock::now();
                uint32_t now_ms
                    = std::chrono::time_point_cast<std::chrono::milliseconds>(now).time_since_epoch().count();

                // receive time from rank zero
                MPI_Bcast(&now_ms, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

                return now_ms;
            }

            uint32_t FromEnvironment::operator()() const
            {
                char* seedStr = nullptr;
                uint32_t seed = 0;
                seedStr = std::getenv("PIC_SEED");
                if(seedStr)
                    seed = std::stoi(seedStr);

                return seed;
            }

        } // namespace seed
    } // namespace random
} // namespace picongpu
