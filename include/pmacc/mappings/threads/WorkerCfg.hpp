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

#include "pmacc/types.hpp"


namespace pmacc
{
    namespace mappings
    {
        namespace threads
        {
            /** holds a worker configuration
             *
             * collection of the compile time number of workers and the runtime worker index
             *
             * @tparam T_numWorkers number of workers which are used to execute this functor
             */
            template<uint32_t T_numWorkers>
            class WorkerCfg
            {
            private:
                //! index of the worker: range [0;T_numWorkers) */
                PMACC_ALIGN(m_workerIdx, uint32_t const);

            public:
                //! number of workers
                static constexpr uint32_t numWorkers = T_numWorkers;

                /** constructor
                 *
                 * @param workerIdx worker index
                 */
                HDINLINE WorkerCfg(uint32_t const workerIdx) : m_workerIdx(workerIdx)
                {
                }

                /** get the worker index
                 *
                 * @return index of the worker
                 */
                HDINLINE uint32_t getWorkerIdx() const
                {
                    return m_workerIdx;
                }

                /** get the number of workers
                 *
                 * @return number of workers
                 */
                HDINLINE static constexpr uint32_t getNumWorkers()
                {
                    return T_numWorkers;
                }
            };

        } // namespace threads
    } // namespace mappings
} // namespace pmacc
