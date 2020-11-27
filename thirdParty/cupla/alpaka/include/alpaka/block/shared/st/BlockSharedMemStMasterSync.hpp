/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/block/shared/st/Traits.hpp>
#include <alpaka/core/AlignedAlloc.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Vectorize.hpp>

#include <functional>
#include <memory>
#include <vector>

namespace alpaka
{
    //#############################################################################
    //! The block shared memory allocator allocating memory with synchronization on the master thread.
    class BlockSharedMemStMasterSync : public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStMasterSync>
    {
    public:
        //-----------------------------------------------------------------------------
        BlockSharedMemStMasterSync(std::function<void()> fnSync, std::function<bool()> fnIsMasterThread)
            : m_syncFn(fnSync)
            , m_isMasterThreadFn(fnIsMasterThread)
        {
        }
        //-----------------------------------------------------------------------------
        BlockSharedMemStMasterSync(BlockSharedMemStMasterSync const&) = delete;
        //-----------------------------------------------------------------------------
        BlockSharedMemStMasterSync(BlockSharedMemStMasterSync&&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(BlockSharedMemStMasterSync const&) -> BlockSharedMemStMasterSync& = delete;
        //-----------------------------------------------------------------------------
        auto operator=(BlockSharedMemStMasterSync&&) -> BlockSharedMemStMasterSync& = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~BlockSharedMemStMasterSync() = default;

    public:
        // TODO: We should add the size of the (current) allocation.
        // This would allow to assert that all parallel function calls request to allocate the same size.
        std::vector<std::unique_ptr<uint8_t, core::AlignedDelete>> mutable m_sharedAllocs;

        std::function<void()> m_syncFn;
        std::function<bool()> m_isMasterThreadFn;
    };

    namespace traits
    {
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
        "-Wcast-align" // "cast from 'unsigned char*' to 'unsigned int*' increases required alignment of target type"
#endif
        //#############################################################################
        template<typename T, std::size_t TuniqueId>
        struct DeclareSharedVar<T, TuniqueId, BlockSharedMemStMasterSync>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto declareVar(BlockSharedMemStMasterSync const& blockSharedMemSt) -> T&
            {
                constexpr std::size_t alignmentInBytes = std::max(core::vectorization::defaultAlignment, alignof(T));

                // Assure that all threads have executed the return of the last allocBlockSharedArr function (if there
                // was one before).
                blockSharedMemSt.m_syncFn();

                if(blockSharedMemSt.m_isMasterThreadFn())
                {
                    blockSharedMemSt.m_sharedAllocs.emplace_back(
                        reinterpret_cast<uint8_t*>(core::alignedAlloc(alignmentInBytes, sizeof(T))));
                }
                blockSharedMemSt.m_syncFn();

                return std::ref(*reinterpret_cast<T*>(blockSharedMemSt.m_sharedAllocs.back().get()));
            }
        };
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
        //#############################################################################
        template<>
        struct FreeSharedVars<BlockSharedMemStMasterSync>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto freeVars(BlockSharedMemStMasterSync const& blockSharedMemSt) -> void
            {
                blockSharedMemSt.m_sharedAllocs.clear();
            }
        };
    } // namespace traits
} // namespace alpaka
