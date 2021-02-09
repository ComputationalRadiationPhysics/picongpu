/* Copyright 2017-2021 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/mappings/threads/IdxConfig.hpp"
#include "pmacc/types.hpp"

#include <utility>


namespace pmacc
{
    namespace mappings
    {
        namespace threads
        {
            /** execute a functor for each index
             *
             * Distribute the indices even over all worker and execute a user defined functor.
             * There is no guarantee in which order the indices will be processed.
             *
             * @tparam T_IdxConfig index domain description
             */
            template<typename T_IdxConfig>
            struct ForEachIdx : public T_IdxConfig
            {
                using T_IdxConfig::domainSize;
                using T_IdxConfig::numCollIter;
                using T_IdxConfig::simdSize;
                using T_IdxConfig::workerSize;

                uint32_t const m_workerIdx;

                static constexpr bool outerLoopCondition
                    = (domainSize % (simdSize * workerSize)) == 0u || (simdSize * workerSize == 1u);

                static constexpr bool innerLoopCondition = (domainSize % simdSize) == 0u || (simdSize == 1u);

                /** constructor
                 *
                 * @param workerIdx index of the worker: range [0;workerSize)
                 */
                HDINLINE
                ForEachIdx(uint32_t const workerIdx) : m_workerIdx(workerIdx)
                {
                }

                /** execute a functor
                 *
                 * @param functor is called for each index which is mapped to the worker
                 *
                 * The functor must fulfill the following interface:
                 * @code
                 * template< typename ... T_Args >
                 * void operator()( uint32_t const linearIdx, uint32_t const idx, T_Args && ... );
                 * @endcode
                 *
                 * @{
                 */
                template<typename T_Functor, typename... T_Args>
                HDINLINE void operator()(T_Functor&& functor, T_Args&&... args) const
                {
                    for(uint32_t i = 0u; i < numCollIter; ++i)
                    {
                        uint32_t const beginWorker = i * simdSize;
                        uint32_t const beginIdx = beginWorker * workerSize + simdSize * m_workerIdx;
                        if(outerLoopCondition || !innerLoopCondition || beginIdx < domainSize)
                        {
                            for(uint32_t j = 0u; j < simdSize; ++j)
                            {
                                uint32_t const localIdx = beginIdx + j;
                                if(innerLoopCondition || localIdx < domainSize)
                                    functor(localIdx, beginWorker + j, std::forward<T_Args>(args)...);
                            }
                        }
                    }
                }

                /** @} */
            };

        } // namespace threads
    } // namespace mappings
} // namespace pmacc
