/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    namespace lockstep
    {
        /** describe a constant index domain
         *
         * describe the size of the index domain and the number of workers to operate on a lockstep domain
         *
         * @tparam T_domainSize number of indices in the domain
         * @tparam T_numWorkers number of worker working on @p T_domainSize
         * @tparam T_simdSize SIMD width
         */
        template<uint32_t T_domainSize, uint32_t T_numWorkers, uint32_t T_simdSize>
        struct Config
        {
            /** number of indices within the domain */
            static constexpr uint32_t domainSize = T_domainSize;
            /** number of worker (threads) working on @p domainSize */
            static constexpr uint32_t numWorkers = T_numWorkers;
            /** SIMD width */
            static constexpr uint32_t simdSize = T_simdSize;

            /** maximum number of indices a worker must process if the domain is equally distributed over all worker */
            static constexpr uint32_t maxIndicesPerWorker
                = ((domainSize + simdSize * numWorkers - 1u) / (simdSize * numWorkers)) * simdSize;
        };
    } // namespace lockstep
} // namespace pmacc
