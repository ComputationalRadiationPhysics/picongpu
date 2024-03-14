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
        /** Helper to manage visibility for methods in BlockCfg.
         *
         * This object provides an optimized access to the one dimensional worker index of BlockCfg.
         * This indirection avoids that the user is accessing the optimized method from a kernel which is not
         * guaranteed launched with a one dimensional block size.
         */
        struct BlockCfgAssume1DBlock
        {
            /** Get the lockstep worker index.
             *
             * @attention This method should only be called if it is guaranteed that the kernel is started with a one
             * dimension block size. In general this method should only be used from
             * lockstep::exec::detail::LockStepKernel.
             *
             * @tparam T_BlockCfg lockstep block configuration
             * @tparam T_Acc alpaka accelerator type
             * @param acc alpaka accelerator
             * @return worker index
             */
            template<typename T_BlockCfg, typename T_Acc>
            HDINLINE static auto getWorker(T_Acc const& acc)
            {
                return T_BlockCfg::getWorkerAssume1DThreads(acc);
            }
        };
    } // namespace detail
    /** Configuration of worker used for a lockstep kernel
     *
     * @tparam T_BlockSize Compile time block size of type pmacc::math::CT::Vector
     *
     * @attention: The real number of workers used for the lockstep kernel depends on the alpaka backend and will
     * be adjusted by this class via the trait pmacc::traits::GetNumWorkers.
     */
    template<typename T_BlockSize>
    struct BlockCfg
    {
        friend struct detail::BlockCfgAssume1DBlock;

        using BlockDomSizeND = T_BlockSize;

        static constexpr uint32_t dim()
        {
            return BlockDomSizeND::dim;
        }

        //! number of workers operating collectively within the block domain
        static constexpr uint32_t blockDomSize()
        {
            return pmacc::math::CT::volume<BlockDomSizeND>::type::value;
        }

        /** adjusted number of workers
         *
         * This number is taking the block size restriction of the alpaka backend into account.
         *
         * @return number of workers
         */
        static constexpr uint32_t numWorkers()
        {
            return pmacc::traits::GetNumWorkers<blockDomSize()>::value;
        }

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
            ALPAKA_ASSERT_ACC(blockExtent.prod() == numWorkers());

            auto const linearThreadIdx = alpaka::mapIdx<1u>(localThreadIdx, blockExtent)[0];
            return Worker<T_Acc, BlockCfg>(acc, linearThreadIdx);
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
            ALPAKA_ASSERT_ACC(blockDim == numWorkers());

            return Worker<T_Acc, BlockCfg>(acc, device::getThreadIdx(acc).x());
        }
    };

    namespace traits
    {
        /** Factory to create a worker config from a PMacc compile time vector.
         *
         * @tparam T_BlockSizeConcept
         * @treturn ::type block configuration type
         */
        template<typename T_BlockSizeConcept, typename T_Sfinae = void>
        struct MakeBlockCfg : std::false_type
        {
        };

        template<typename T_X, typename T_Y, typename T_Z>
        struct MakeBlockCfg<math::CT::Vector<T_X, T_Y, T_Z>> : std::true_type
        {
            using Size = math::CT::Vector<T_X, T_Y, T_Z>;
            using type = BlockCfg<Size>;
        };

        template<typename T_BlockSizeConcept>
        struct MakeBlockCfg<T_BlockSizeConcept, std::void_t<typename T_BlockSizeConcept::BlockDomSizeND>>
            : std::true_type
        {
            using type = typename MakeBlockCfg<typename T_BlockSizeConcept::BlockDomSizeND>::type;
        };

        template<typename T_BlockSizeConcept>
        struct MakeBlockCfg<T_BlockSizeConcept, std::enable_if_t<T_BlockSizeConcept::blockDomSize != 0>>
            : std::true_type
        {
            using type = BlockCfg<math::CT::UInt32<T_BlockSizeConcept::blockDomSize>>;
        };

        //! Factory alias to get the block configuration type.
        template<typename T_BlockSizeConcept>
        using MakeBlockCfg_t = typename MakeBlockCfg<T_BlockSizeConcept>::type;

        template<typename T_BlockSizeConcept>
        constexpr bool hasBlockCfg_v = MakeBlockCfg<T_BlockSizeConcept>::value;
    } // namespace traits

    /** Creates a lockstep block configuration.
     *
     * @tparam T_numSuggestedWorkers Suggested number of lockstep workers.
     * @return lockstep block configuration
     */
    template<uint32_t T_numSuggestedWorkers>
    HDINLINE auto makeBlockCfg()
    {
        return BlockCfg<math::CT::UInt32<T_numSuggestedWorkers>>{};
    }

    /** Specialization to create a lockstep block configuration out of a PMacc compile time vector.
     *
     * @tparam T_BlockSizeConcept Type where the trait MakeBlockCfg is specialized for.
     * @return lockstep block configuration
     */
    template<typename T_BlockSizeConcept>
    HDINLINE auto makeBlockCfg(T_BlockSizeConcept const&)
    {
        return traits::MakeBlockCfg_t<T_BlockSizeConcept>{};
    }

} // namespace pmacc::lockstep
