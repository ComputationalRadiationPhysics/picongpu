/*
 * SPDX-FileCopyrightText: Axel Huebl, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/algorithms/reverseBits.hpp"
#include "pmacc/types.hpp"

#include <limits>

namespace pmacc
{
    namespace mpi
    {
        /** Calculate a Seed per Rank
         *
         * This functor derives a unique seed for each MPI rank (or GPU) from
         * a given global seed in a deterministic manner.
         *
         * @tparam T_DIM Dimensionality of the simulation (1-3 D)
         */
        template<unsigned T_DIM>
        struct SeedPerRank
        {
            /** Functor implementation
             *
             * This method provides a guaranteed unique number per MPI rank
             * (or GPU). When a (only locally unique) localSeed parameter is used
             * it is furthermore guaranteed that this number does not collide
             * with an other seed.
             *
             * @param localSeed Initial seed to vary two identical simulations
             *                  can have been xor'ed with e.g. a unique species id
             *                  to get an unique seed per species
             * @return uint32_t seed
             */
            uint32_t operator()(uint32_t localSeed)
            {
                auto& gc = pmacc::Environment<T_DIM>::get().GridController();

                uint32_t rank = gc.getGlobalRank();
                /* We put the rank into the upper bits to allow values which start
                 * from zero (e.g. cellIdxs, time steps) to be used as additional seed contributors
                 * Those would then write to the lower bits leaving the upper bits alone
                 * which still results in globally unique seeds
                 */
                uint32_t globalUniqueSeed = reverseBits(rank);
                /* localSeed often contains a counted number, so we rotate it by some bits to not "destroy"
                 * the counted rank that is already there. Also it is not reversed to get a different pattern
                 */
                localSeed = (localSeed << 16) | (localSeed >> (sizeof(uint32_t) * CHAR_BIT - 16));
                globalUniqueSeed ^= localSeed;
                /* For any globally constant localSeed globalUniqueSeed is now guaranteed
                 * to be globally unique
                 */
                return globalUniqueSeed;
            }
        };

    } /* namespace mpi */
} // namespace pmacc
