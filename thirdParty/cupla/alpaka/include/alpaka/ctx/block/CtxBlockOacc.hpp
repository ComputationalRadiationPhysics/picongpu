/* Copyright 2022 Jeffrey Kelling, Bernhard Manfred Gruber
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#    if _OPENACC < 201306
#        error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC 2.0 or higher!
#    endif

// Base classes.
#    include <alpaka/atomic/AtomicOaccExtended.hpp>
#    include <alpaka/block/shared/dyn/BlockSharedMemDynMember.hpp>
#    include <alpaka/block/shared/st/BlockSharedMemStMember.hpp>
#    include <alpaka/block/sync/BlockSyncBarrierOacc.hpp>
#    include <alpaka/idx/gb/IdxGbLinear.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

// Specialized traits.
#    include <alpaka/idx/Traits.hpp>

#    include <limits>
#    include <typeinfo>

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelOacc;

    //! The OpenACC block context.
    template<typename TDim, typename TIdx>
    class CtxBlockOacc final
        : public WorkDivMembers<TDim, TIdx>
        , public gb::IdxGbLinear<TDim, TIdx>
        , public AtomicOaccExtended<hierarchy::Grids> // grid atomics
        , public AtomicOaccExtended<hierarchy::Blocks> // block atomics
        , public AtomicOaccExtended<hierarchy::Threads> // thread atomics
        , public BlockSharedMemDynMember<>
        , public detail::BlockSharedMemStMemberImpl<4>
        , public BlockSyncBarrierOacc
        , public concepts::Implements<ConceptBlockSharedSt, CtxBlockOacc<TDim, TIdx>>
    {
    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::alpaka::TaskKernelOacc;

    protected:
        CtxBlockOacc(
            Vec<TDim, TIdx> const& gridBlockExtent,
            Vec<TDim, TIdx> const& blockThreadExtent,
            Vec<TDim, TIdx> const& threadElemExtent,
            TIdx const& gridBlockIdx,
            std::size_t const& blockSharedMemDynSizeBytes,
            std::uint32_t* gridsMutex,
            std::uint32_t* blocksMutex)
            : WorkDivMembers<TDim, TIdx>(gridBlockExtent, blockThreadExtent, threadElemExtent)
            , gb::IdxGbLinear<TDim, TIdx>(gridBlockIdx)
            , // grid atomics, using block lock is save as long as only one synchronous queue is used
            AtomicOaccExtended<hierarchy::Grids>(gridsMutex)
            , AtomicOaccExtended<hierarchy::Blocks>(blocksMutex)
            , AtomicOaccExtended<hierarchy::Threads>()
            , BlockSharedMemDynMember<>(blockSharedMemDynSizeBytes)
            ,
            //! \TODO can with some TMP determine the amount of statically alloced smem from the kernelFuncObj?
            detail::BlockSharedMemStMemberImpl<4>(staticMemBegin(), staticMemCapacity())
            , BlockSyncBarrierOacc()
        {
        }
    };

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct SyncBlockThreads<CtxBlockOacc<TDim, TIdx>>
        {
            //! Execute op with single thread (any idx, last thread to
            //! arrive at barrier executes) syncing before and after
            template<typename TOp>
            ALPAKA_FN_HOST static auto masterOpBlockThreads(CtxBlockOacc<TDim, TIdx> const& acc, TOp&& op) -> void
            {
                auto const slot = (acc.m_generation & 1) << 1;
                int const workerNum = static_cast<int>(getWorkDiv<Block, Threads>(acc).prod());
                int sum;
                // Workaround to use an array in an atomic capture rather than
                // using the data member m_syncCounter array directly.
                // The change is semantically equivalent.
                // However, this should work per the OpenACC standard, but appears to be compiler
                // issue causing a runtime error.  The error was seen the 20.7 release
                // of the NVIDIA HPC Compiler but may be corrected in future releases.
                int* m_syncCounter = acc.m_syncCounter;
#    pragma acc atomic capture
                {
                    ++m_syncCounter[slot];
                    sum = m_syncCounter[slot];
                }
                if(sum == workerNum)
                {
                    ++acc.m_generation;
                    int const nextSlot = (acc.m_generation & 1) << 1;
                    m_syncCounter[nextSlot] = 0;
                    m_syncCounter[nextSlot + 1] = 0;
                    op();
                }
                while(sum < workerNum)
                {
#    pragma acc atomic read
                    sum = m_syncCounter[slot];
                }
#    pragma acc atomic capture
                {
                    ++m_syncCounter[slot + 1];
                    sum = m_syncCounter[slot + 1];
                }
                while(sum < workerNum)
                {
#    pragma acc atomic read
                    sum = m_syncCounter[slot + 1];
                }
            }

            ALPAKA_FN_HOST static auto syncBlockThreads(CtxBlockOacc<TDim, TIdx> const& acc) -> void
            {
                masterOpBlockThreads<>(acc, []() {});
            }
        };

        namespace oacc
        {
            namespace detail
            {
                template<typename TOp>
                struct AtomicOp;
                template<>
                struct AtomicOp<BlockCount>
                {
                    void operator()(int& result, bool value)
                    {
#    pragma acc atomic update
                        result += static_cast<int>(value);
                    }
                };
                template<>
                struct AtomicOp<BlockAnd>
                {
                    void operator()(int& result, bool value)
                    {
#    pragma acc atomic update
                        result &= static_cast<int>(value);
                    }
                };
                template<>
                struct AtomicOp<BlockOr>
                {
                    void operator()(int& result, bool value)
                    {
#    pragma acc atomic update
                        result |= static_cast<int>(value);
                    }
                };
            } // namespace detail
        } // namespace oacc

        template<typename TOp, typename TDim, typename TIdx>
        struct SyncBlockThreadsPredicate<TOp, CtxBlockOacc<TDim, TIdx>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(
                CtxBlockOacc<TDim, TIdx> const& blockSync,
                int predicate) -> int
            {
                // implicit snyc
                SyncBlockThreads<CtxBlockOacc<TDim, TIdx>>::masterOpBlockThreads(
                    blockSync,
                    [&blockSync]() { blockSync.m_result = TOp::InitialValue; });

                int& result(blockSync.m_result);
                bool const predicateBool(predicate != 0);

                oacc::detail::AtomicOp<TOp>()(result, predicateBool);

                SyncBlockThreads<CtxBlockOacc<TDim, TIdx>>::syncBlockThreads(blockSync);

                return blockSync.m_result;
            }
        };

        //! The OpenACC accelerator dimension getter trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<CtxBlockOacc<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The OpenACC accelerator idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<CtxBlockOacc<TDim, TIdx>>
        {
            using type = TIdx;
        };

        template<typename T, typename TDim, typename TIdx, std::size_t TuniqueId>
        struct DeclareSharedVar<T, TuniqueId, CtxBlockOacc<TDim, TIdx>>
        {
            static auto declareVar(CtxBlockOacc<TDim, TIdx> const& smem) -> T&
            {
                auto* data = smem.template getVarPtr<T>(TuniqueId);

                if(!data)
                {
                    trait::SyncBlockThreads<CtxBlockOacc<TDim, TIdx>>::masterOpBlockThreads(
                        smem,
                        [&data, &smem]() { smem.template alloc<T>(TuniqueId); });
                    data = smem.template getLatestVarPtr<T>();
                }
                ALPAKA_ASSERT_OFFLOAD(data != nullptr);
                return *data;
            }
        };

        template<typename TDim, typename TIdx>
        struct FreeSharedVars<CtxBlockOacc<TDim, TIdx>>
        {
            static auto freeVars(CtxBlockOacc<TDim, TIdx> const&) -> void
            {
                // Nothing to do. Block shared memory is automatically freed when all threads left the block.
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
