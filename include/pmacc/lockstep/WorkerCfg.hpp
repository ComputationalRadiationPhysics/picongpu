/* Copyright 2022-2023 Rene Widera
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

#include "pmacc/device/threadInfo.hpp"
#include "pmacc/lockstep/Worker.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"
#include "pmacc/types.hpp"

#include "pmacc/math/vector/compile-time/Vector.hpp"


namespace pmacc::lockstep
{
    namespace detail
    {
        /** Helper to manage visibility for methods in WorkerCfg.
         *
         * This object provides an optimized access to the one dimensional worker index of WorkerCfg.
         * This indirection avoids that the user is accessing the optimized method from a kernel which is not
         * guaranteed launched with a one dimensional block size.
         */
        struct WorkerCfgAssume1DBlock
        {
            /** Get the lockstep worker index.
             *
             * @attention This method should only be called if it is guaranteed that the kernel is started with a one
             * dimension block size. In general this method should only be used from
             * lockstep::exec::detail::LockStepKernel.
             *
             * @tparam T_WorkerCfg lockstep worker configuration
             * @tparam T_Acc alpaka accelerator type
             * @param acc alpaka accelerator
             * @return worker index
             */
            template<typename T_WorkerCfg, typename T_Acc>
            HDINLINE static auto getWorker(T_Acc const& acc)
            {
                return T_WorkerCfg::getWorkerAssume1DThreads(acc);
            }
        };
    } // namespace detail
    /** Configuration of worker used for a lockstep kernel
     *
     * @tparam T_numSuggestedWorkers Suggested number of lockstep workers. Do not assume that the suggested number of
     *                               workers is used within the kernel. The real used number of worker can be queried
     *                               with getNumWorkers() or via the member variable numWorkers.
     *
     * @attention: The real number of workers used for the lockstep kernel depends on the alpaka backend and will
     * be adjusted by this class via the trait pmacc::traits::GetNumWorkers.
     */
    template<uint32_t T_numSuggestedWorkers>
    struct WorkerCfg
    {
        friend struct detail::WorkerCfgAssume1DBlock;

        /** adjusted number of workers
         *
         * This number is taking the block size restriction of the alpaka backend into account.
         */
        static constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<T_numSuggestedWorkers>::value;

        /** get the worker index
         *
         * @return index of the worker
         */
        template<typename T_Acc>
        HDINLINE static auto getWorker(T_Acc const& acc)
        {
            auto const localThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
            auto const blockExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

            // validate that the kernel is started with the correct number of threads
            ALPAKA_ASSERT_ACC(blockExtent.prod() == numWorkers);

            auto const linearThreadIdx = alpaka::mapIdx<1u>(localThreadIdx, blockExtent)[0];
            return Worker<T_Acc, T_numSuggestedWorkers>(acc, linearThreadIdx);
        }

        /** get the number of workers
         *
         * @return number of workers
         */
        HDINLINE static constexpr uint32_t getNumWorkers()
        {
            return numWorkers;
        }

    private:
        /** Get the lockstep worker index.
         *
         * @tparam T_Acc alpaka accelerator type
         * @param acc alpaka accelerator
         * @return lockstep worker index
         */
        template<typename T_Acc>
        HDINLINE static auto getWorkerAssume1DThreads(T_Acc const& acc)
        {
            [[maybe_unused]] auto const blockDim = device::getBlockSize(acc).x();
            // validate that the kernel is started with the correct number of threads
            ALPAKA_ASSERT_ACC(blockDim == numWorkers);

            return Worker<T_Acc, T_numSuggestedWorkers>(acc, device::getThreadIdx(acc).x());
        }
    };

    inline namespace traits
    {
        /** Factory to create a worker config from a PMacc compile time vector.
         *
         * @tparam T_CTVector PMacc compile time vector
         * @treturn ::type worker configuration type
         */
        template<typename T_CTVector>
        struct MakeWorkerCfg;

        template<typename T_X, typename T_Y, typename T_Z>
        struct MakeWorkerCfg<math::CT::Vector<T_X, T_Y, T_Z>>
        {
            using Size = math::CT::Vector<T_X, T_Y, T_Z>;
            using type = WorkerCfg<math::CT::volume<Size>::type::value>;
        };

        //! Factory alias to get the worker configuration type.
        template<typename T_CTVector>
        using MakeWorkerCfg_t = typename MakeWorkerCfg<T_CTVector>::type;
    } // namespace traits

    /** Creates a lockstep worker configuration.
     *
     * @tparam T_numSuggestedWorkers Suggested number of lockstep workers.
     * @return lockstep worker configuration
     */
    template<uint32_t T_numSuggestedWorkers>
    HDINLINE auto makeWorkerCfg()
    {
        return WorkerCfg<T_numSuggestedWorkers>{};
    }

    /** Specialization to create a lockstep worker configuration out of a PMacc compile time vector.
     *
     * @tparam T_X number of elements in X
     * @tparam T_Y number of elements in Y
     * @tparam T_Z number of elements in Z
     * @return lockstep worker configuration
     */
    template<typename T_X, typename T_Y, typename T_Z>
    HDINLINE auto makeWorkerCfg(math::CT::Vector<T_X, T_Y, T_Z> const&)
    {
        using Size = math::CT::Vector<T_X, T_Y, T_Z>;
        return MakeWorkerCfg_t<Size>{};
    }
} // namespace pmacc::lockstep
