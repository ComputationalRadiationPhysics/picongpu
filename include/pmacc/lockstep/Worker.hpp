/* Copyright 2017-2023 Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/device/threadInfo.hpp"
#include "pmacc/types.hpp"


namespace pmacc::lockstep
{
    template<typename T_BlockSize>
    struct BlockCfg;

    /** Entity of an worker.
     *
     * Context object used for lockstep programming. This object is providing access to the alpaka accelerator and
     * indicies used for the lockstep programming model.
     *
     * @tparam T_numSuggestedWorkers Suggested number of lockstep workers. Do not assume that the suggested number of
     *                               workers is used within the kernel. The real used number of worker can be queried
     *                               with numWorkers() or via the member variable numWorkers.
     */
    template<typename T_Acc, typename T_BlockCfg>
    class Worker
    {
    private:
        //! index of the worker: range [0;numWOrkers) */
        PMACC_ALIGN(m_workerIdx, uint32_t const);
        T_Acc const& m_acc;

        template<typename T_BlockConfig>
        friend struct BlockCfg;

        /** constructor
         *
         * @param workerIdx worker index
         */
        HDINLINE Worker(T_Acc const& acc, uint32_t const workerIdx) : m_workerIdx(std::move(workerIdx)), m_acc(acc)
        {
        }

    public:
        Worker(Worker const&) = default;

        Worker& operator=(Worker const&) = delete;

        using Acc = T_Acc;

        /** get the alpaka accelerator
         *
         * @return alpaka accelerator
         */
        HDINLINE auto& getAcc() const
        {
            return m_acc;
        }

        /** number of workers operating collectively within the block domain
         *
         * @return number of workers
         */
        static constexpr uint32_t numWorkers()
        {
            return T_BlockCfg::numWorkers();
        }

        /** N-dimensional block domain size type (math::CT::Vector<>)
         */
        using BlockDomSizeND = typename T_BlockCfg::BlockDomSizeND;

        /** number of indices in the block domain
         */
        static constexpr uint32_t blockDomSize()
        {
            return T_BlockCfg::blockDomSize();
        }

        /** N-dimensional block domain size
         *
         * @return instance of math::Vector<>
         */
        HDINLINE auto blockDomSizeND() const
        {
            return BlockDomSizeND::toRT();
        };

        /** N-dimensional grid domain size
         *
         * @return instance of math::Vector<>
         */
        HDINLINE auto gridDomSizeND() const
        {
            return device::getGridSize(getAcc());
        }

        /** number of blocks in the grid domain
         */
        HDINLINE auto gridDomSize() const
        {
            return gridDomSizeND().productOfComponents();
        }

        /** N-dimensional index of a block
         *
         * @return instance of math::Vector<>
         */
        HDINLINE auto blockDomIdxND() const
        {
            return device::getBlockIdx(getAcc());
        }

        /** index of blocks in the grid domain
         */
        HDINLINE auto blockDomIdx() const
        {
            return pmacc::math::linearize(gridDomSizeND(), blockDomIdxND());
        }

        /** synchronize all workers
         *
         * @attention It is not allowed to call this method inside of an if branch or a loop if it is not
         * guaranteed that all workers are executing the same code branch.
         */
        HDINLINE void sync() const
        {
            alpaka::syncBlockThreads(m_acc);
        }

        /** get the worker index
         *
         * @return index of the worker
         */
        HDINLINE uint32_t workerIdx() const
        {
            return m_workerIdx;
        }
    };
} // namespace pmacc::lockstep
