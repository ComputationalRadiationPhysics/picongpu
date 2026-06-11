/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "picongpu/random/seed/Seed.hpp"

#include <chrono>
#include <cstdlib>
#include <string>

#include <mpi.h>

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
