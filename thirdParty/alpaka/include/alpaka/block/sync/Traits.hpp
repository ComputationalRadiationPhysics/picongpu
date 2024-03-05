/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"

#include <type_traits>

namespace alpaka
{
    struct ConceptBlockSync
    {
    };

    //! The block synchronization traits.
    namespace trait
    {
        //! The block synchronization operation trait.
        template<typename TBlockSync, typename TSfinae = void>
        struct SyncBlockThreads;

        //! The block synchronization and predicate operation trait.
        template<typename TOp, typename TBlockSync, typename TSfinae = void>
        struct SyncBlockThreadsPredicate;
    } // namespace trait

    //! Synchronizes all threads within the current block (independently for all blocks).
    //!
    //! \tparam TBlockSync The block synchronization implementation type.
    //! \param blockSync The block synchronization implementation.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TBlockSync>
    ALPAKA_FN_ACC auto syncBlockThreads(TBlockSync const& blockSync) -> void
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptBlockSync, TBlockSync>;
        trait::SyncBlockThreads<ImplementationBase>::syncBlockThreads(blockSync);
    }

    //! The counting function object.
    struct BlockCount
    {
        enum
        {
            InitialValue = 0u
        };

        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T>
        ALPAKA_FN_HOST_ACC auto operator()(T const& currentResult, T const& value) const -> T
        {
            return currentResult + static_cast<T>(value != static_cast<T>(0));
        }
    };

    //! The logical and function object.
    struct BlockAnd
    {
        enum
        {
            InitialValue = 1u
        };

        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T>
        ALPAKA_FN_HOST_ACC auto operator()(T const& currentResult, T const& value) const -> T
        {
            return static_cast<T>(currentResult && (value != static_cast<T>(0)));
        }
    };

    //! The logical or function object.
    struct BlockOr
    {
        enum
        {
            InitialValue = 0u
        };

        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T>
        ALPAKA_FN_HOST_ACC auto operator()(T const& currentResult, T const& value) const -> T
        {
            return static_cast<T>(currentResult || (value != static_cast<T>(0)));
        }
    };

    //! Synchronizes all threads within the current block (independently for all blocks),
    //! evaluates the predicate for all threads and returns the combination of all the results
    //! computed via TOp.
    //!
    //! \tparam TOp The operation used to combine the predicate values of all threads.
    //! \tparam TBlockSync The block synchronization implementation type.
    //! \param blockSync The block synchronization implementation.
    //! \param predicate The predicate value of the current thread.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOp, typename TBlockSync>
    ALPAKA_FN_ACC auto syncBlockThreadsPredicate(TBlockSync const& blockSync, int predicate) -> int
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptBlockSync, TBlockSync>;
        return trait::SyncBlockThreadsPredicate<TOp, ImplementationBase>::syncBlockThreadsPredicate(
            blockSync,
            predicate);
    }
} // namespace alpaka
