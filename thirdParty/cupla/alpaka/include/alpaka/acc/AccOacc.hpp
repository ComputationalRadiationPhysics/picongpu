/* Copyright 2022 Jeffrey Kelling, Jan Stephan, Bernhard Manfred Gruber
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
#    include <alpaka/core/DemangleTypeNames.hpp>
#    include <alpaka/ctx/block/CtxBlockOacc.hpp>
#    include <alpaka/idx/bt/IdxBtLinear.hpp>
#    include <alpaka/intrinsic/IntrinsicFallback.hpp>
#    include <alpaka/math/MathStdLib.hpp>
#    include <alpaka/mem/fence/MemFenceOacc.hpp>
#    include <alpaka/rand/RandDefault.hpp>
#    include <alpaka/warp/WarpSingleThread.hpp>

// Specialized traits.
#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

// Implementation details.
#    include <alpaka/acc/Tag.hpp>
#    include <alpaka/core/ClipCast.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/ctx/block/CtxBlockOacc.hpp>
#    include <alpaka/dev/DevOacc.hpp>

#    include <limits>
#    include <typeinfo>

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelOacc;

    // define max gang/worker num because there is no standart way in OpenACC to
    // get this information
#    ifndef ALPAKA_OACC_MAX_GANG_NUM
    constexpr size_t oaccMaxGangNum = 1;
#    else
    constexpr size_t oaccMaxGangNum = ALPAKA_OACC_MAX_GANG_NUM;
#    endif
#    if defined(ALPAKA_OFFLOAD_MAX_BLOCK_SIZE) && ALPAKA_OFFLOAD_MAX_BLOCK_SIZE > 0
    constexpr size_t oaccMaxWorkerNum = ALPAKA_OFFLOAD_MAX_BLOCK_SIZE;
#    else
    constexpr size_t oaccMaxWorkerNum = 1;
#    endif

    //! The OpenACC accelerator.
    template<typename TDim, typename TIdx>
    class AccOacc final
        : public bt::IdxBtLinear<TDim, TIdx>
        , public math::MathStdLib
        , public MemFenceOacc
        , public rand::RandDefault
        , public warp::WarpSingleThread
        ,
          // NVHPC calls a builtin in the STL implementation, which fails in OpenACC offload, using fallback
          public IntrinsicFallback
        , public concepts::Implements<ConceptAcc, AccOacc<TDim, TIdx>>
        , public concepts::Implements<ConceptWorkDiv, AccOacc<TDim, TIdx>>
        , public concepts::Implements<ConceptBlockSharedDyn, AccOacc<TDim, TIdx>>
        , public concepts::Implements<ConceptBlockSharedSt, AccOacc<TDim, TIdx>>
        , public concepts::Implements<ConceptBlockSync, AccOacc<TDim, TIdx>>
        , public concepts::Implements<ConceptIdxGb, AccOacc<TDim, TIdx>>
        , public concepts::Implements<ConceptAtomicGrids, AccOacc<TDim, TIdx>>
        , public concepts::Implements<ConceptAtomicBlocks, AccOacc<TDim, TIdx>>
        , public concepts::Implements<ConceptAtomicThreads, AccOacc<TDim, TIdx>>
    {
        template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::alpaka::TaskKernelOacc;

    protected:
        AccOacc(TIdx const& blockThreadIdx, CtxBlockOacc<TDim, TIdx>& blockShared)
            : bt::IdxBtLinear<TDim, TIdx>(blockThreadIdx)
            , math::MathStdLib()
            , MemFenceOacc()
            , rand::RandDefault()
            , m_blockShared(blockShared)
        {
        }

    public:
        AccOacc(AccOacc const&) = delete;
        AccOacc(AccOacc&&) = delete;
        auto operator=(AccOacc const&) -> AccOacc& = delete;
        auto operator=(AccOacc&&) -> AccOacc& = delete;

        CtxBlockOacc<TDim, TIdx>& m_blockShared;
    };

    namespace trait
    {
        //! The OpenACC accelerator accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccOacc<TDim, TIdx>>
        {
            using type = AccOacc<TDim, TIdx>;
        };
        //! The OpenACC accelerator device properties get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccDevProps<AccOacc<TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getAccDevProps(DevOacc const& /* dev */) -> AccDevProps<TDim, TIdx>
            {
                auto const blockThreadCountMax(alpaka::core::clipCast<TIdx>(oaccMaxWorkerNum));
                auto const gridBlockCountMax(alpaka::core::clipCast<TIdx>(oaccMaxGangNum));

                return {// m_multiProcessorCount
                        static_cast<TIdx>(gridBlockCountMax), // TODO: standardize a way to return "unknown"
                                                              // m_gridBlockExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        Vec<TDim, TIdx>::all(blockThreadCountMax),
                        // m_blockThreadCountMax
                        blockThreadCountMax,
                        // m_threadElemExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        CtxBlockOacc<TDim, TIdx>::staticAllocBytes()};
            }
        };
        //! The OpenACC accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccOacc<TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return "AccOacc<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
            }
        };

        //! The OpenACC accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccOacc<TDim, TIdx>>
        {
            using type = DevOacc;
        };

        //! The OpenACC accelerator dimension getter trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<AccOacc<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The OpenACC accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccOacc<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            ALPAKA_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                return TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };

        //! The OpenACC execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PltfType<AccOacc<TDim, TIdx>>
        {
            using type = PltfOacc;
        };

        //! The OpenACC accelerator idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<AccOacc<TDim, TIdx>>
        {
            using type = TIdx;
        };

        //! The OpenACC accelerator grid block index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<AccOacc<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //! \return The index of the current block in the grid.
            template<typename TWorkDiv>
            static auto getIdx(AccOacc<TDim, TIdx> const& idx, TWorkDiv const& workDiv) -> Vec<TDim, TIdx>
            {
                // // \TODO: Would it be faster to precompute the index and cache it inside an array?
                return mapIdx<TDim::value>(
                    Vec<DimInt<1u>, TIdx>(idx.m_blockShared.m_gridBlockIdx),
                    getWorkDiv<Grid, Blocks>(workDiv));
            }
        };

        template<typename TIdx>
        struct GetIdx<AccOacc<DimInt<1u>, TIdx>, origin::Grid, unit::Blocks>
        {
            //! \return The index of the current block in the grid.
            template<typename TWorkDiv>
            static auto getIdx(AccOacc<DimInt<1u>, TIdx> const& idx, TWorkDiv const&) -> Vec<DimInt<1u>, TIdx>
            {
                return idx.m_blockShared.m_gridBlockIdx;
            }
        };

        template<typename T, typename TDim, typename TIdx>
        struct GetDynSharedMem<T, AccOacc<TDim, TIdx>>
        {
#    if BOOST_COMP_GNUC
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wcast-align" // "cast from 'unsigned char*' to 'unsigned int*' increases
                                                      // required alignment of target type"
#    endif
            static auto getMem(AccOacc<TDim, TIdx> const& mem) -> T*
            {
                return reinterpret_cast<T*>(mem.m_blockShared.dynMemBegin());
            }
#    if BOOST_COMP_GNUC
#        pragma GCC diagnostic pop
#    endif
        };

        template<typename T, typename TDim, typename TIdx, std::size_t TuniqueId>
        struct DeclareSharedVar<T, TuniqueId, AccOacc<TDim, TIdx>>
        {
            static auto declareVar(AccOacc<TDim, TIdx> const& smem) -> T&
            {
                return alpaka::declareSharedVar<T, TuniqueId>(smem.m_blockShared);
            }
        };

        template<typename TDim, typename TIdx>
        struct FreeSharedVars<AccOacc<TDim, TIdx>>
        {
            static auto freeVars(AccOacc<TDim, TIdx> const& smem) -> void
            {
                alpaka::freeSharedVars(smem.m_blockShared);
            }
        };

        template<typename TDim, typename TIdx>
        struct SyncBlockThreads<AccOacc<TDim, TIdx>>
        {
            //! Execute op with single thread (any idx, last thread to
            //! arrive at barrier executes) syncing before and after
            template<typename TOp>
            ALPAKA_FN_HOST static auto masterOpBlockThreads(AccOacc<TDim, TIdx> const& acc, TOp&& op) -> void
            {
                SyncBlockThreads<CtxBlockOacc<TDim, TIdx>>::masterOpBlockThreads(acc.m_blockShared, op);
            }

            ALPAKA_FN_HOST static auto syncBlockThreads(AccOacc<TDim, TIdx> const& acc) -> void
            {
                SyncBlockThreads<CtxBlockOacc<TDim, TIdx>>::syncBlockThreads(acc.m_blockShared);
            }
        };

        template<typename TOp, typename TDim, typename TIdx>
        struct SyncBlockThreadsPredicate<TOp, AccOacc<TDim, TIdx>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(AccOacc<TDim, TIdx> const& acc, int predicate) -> int
            {
                return SyncBlockThreadsPredicate<TOp, CtxBlockOacc<TDim, TIdx>>::syncBlockThreadsPredicate(
                    acc.m_blockShared,
                    predicate);
            }
        };

        //! The OpenACC atomicOp trait specialization.
        template<typename TDim, typename TIdx, typename TOp, typename T, typename THierarchy>
        struct AtomicOp<TOp, AccOacc<TDim, TIdx>, T, THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC static auto atomicOp(AccOacc<TDim, TIdx> const& acc, T* const addr, T const& value) -> T
            {
                return AtomicOp<TOp, AtomicOaccExtended<THierarchy>, T, THierarchy>::atomicOp(
                    acc.m_blockShared,
                    addr,
                    value);
            }

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AccOacc<TDim, TIdx> const& acc,
                T* const addr,
                T const& compare,
                T const& value) -> T
            {
                return AtomicOp<TOp, AtomicOaccExtended<THierarchy>, T, THierarchy>::atomicOp(
                    acc.m_blockShared,
                    addr,
                    compare,
                    value);
            }
        };

        //! The OpenACC grid block extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<AccOacc<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //! \return The number of blocks in each dimension of the grid.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getWorkDiv(AccOacc<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
            {
                return GetWorkDiv<WorkDivMembers<TDim, TIdx>, origin::Grid, unit::Blocks>::getWorkDiv(
                    workDiv.m_blockShared);
            }
        };

        //! The OpenACC block thread extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<AccOacc<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The number of threads in each dimension of a block.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getWorkDiv(AccOacc<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
            {
                return GetWorkDiv<WorkDivMembers<TDim, TIdx>, origin::Block, unit::Threads>::getWorkDiv(
                    workDiv.m_blockShared);
            }
        };

        //! The OpenACC thread element extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<AccOacc<TDim, TIdx>, origin::Thread, unit::Elems>
        {
            //! \return The number of elements in each dimension of a thread.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getWorkDiv(AccOacc<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
            {
                return GetWorkDiv<WorkDivMembers<TDim, TIdx>, origin::Thread, unit::Elems>::getWorkDiv(
                    workDiv.m_blockShared);
            }
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccOacc<TDim, TIdx>>
        {
            using type = alpaka::TagOacc;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagOacc, TDim, TIdx>
        {
            using type = alpaka::AccOacc<TDim, TIdx>;
        };
    } // namespace trait
} // namespace alpaka

#endif
