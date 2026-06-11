/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/algorithms/Set.hpp"
#include "picongpu/defines.hpp"

#include <pmacc/mappings/threads/ThreadCollective.hpp>
#include <pmacc/memory/boxes/CachedBox.hpp>
#include <pmacc/types.hpp>

namespace picongpu
{
    namespace currentSolver
    {
        namespace detail
        {
            /** Transparent cache implementation for the current solver
             *
             * @tparam T_Strategy Used strategy to reduce the scattered data [currentSolver::strategy]
             * @tparam T_Sfinae Optional specialization
             */
            template<typename T_Strategy, typename T_Sfinae = void>
            struct Cache;

            template<typename T_Strategy>
            struct Cache<T_Strategy, typename std::enable_if<T_Strategy::useBlockCache>::type>
            {
                /** Create a cache
                 *
                 * @attention thread-collective operation, requires external thread synchronization
                 */
                template<typename T_BlockDescription, typename T_Worker, typename T_FieldBox>
                DINLINE static auto create(T_Worker const& worker, T_FieldBox const& fieldBox)
#if (!BOOST_COMP_CLANG)
                    -> decltype(pmacc::CachedBox::create<0u, typename T_FieldBox::ValueType>(
                        worker,
                        std::declval<T_BlockDescription>()))
#endif
                {
                    using ValueType = typename T_FieldBox::ValueType;
                    /* this memory is used by all virtual blocks */
                    auto cache = CachedBox::create<0u, ValueType>(worker, T_BlockDescription{});

                    Set<ValueType> set(ValueType::create(0.0_X));
                    auto collectiveFill = makeThreadCollective<T_BlockDescription>();

                    /* initialize shared memory with zeros */
                    collectiveFill(worker, set, cache);
                    return cache;
                }

                /** Flush the cache
                 *
                 * @attention thread-collective operation, requires external thread synchronization
                 */
                template<typename T_BlockDescription, typename T_Worker, typename T_FieldBox, typename T_FieldCache>
                DINLINE static void flush(T_Worker const& worker, T_FieldBox fieldBox, T_FieldCache const& cachedBox)
                {
                    typename T_Strategy::GridReductionOp const op;
                    auto collectiveFlush = makeThreadCollective<T_BlockDescription>();

                    /* write scatter results back to the global memory */
                    collectiveFlush(worker, op, fieldBox, cachedBox);
                }
            };

            template<typename T_Strategy>
            struct Cache<T_Strategy, typename std::enable_if<!T_Strategy::useBlockCache>::type>
            {
                /** Create a cache
                 *
                 * @attention thread-collective operation, requires external thread synchronization
                 */
                template<typename T_BlockDescription, typename T_Worker, typename T_FieldBox>
                DINLINE static auto create([[maybe_unused]] T_Worker const& worker, T_FieldBox const& fieldBox)
#if (!BOOST_COMP_CLANG)
                    -> T_FieldBox
#endif
                {
                    return fieldBox;
                }

                /** Flush the cache
                 *
                 * @attention thread-collective operation, requires external thread synchronization
                 */
                template<typename T_BlockDescription, typename T_Worker, typename T_FieldBox, typename T_FieldCache>
                DINLINE static void flush(
                    T_Worker const& /*worker*/,
                    T_FieldBox /*fieldBox*/,
                    T_FieldCache const& /*cachedBox*/)
                {
                }
            };
        } // namespace detail
    } // namespace currentSolver
} // namespace picongpu
