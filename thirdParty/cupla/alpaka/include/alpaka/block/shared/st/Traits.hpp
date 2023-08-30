/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"

#include <type_traits>

namespace alpaka
{
    struct ConceptBlockSharedSt
    {
    };

    //! The block shared static memory operation trait.
    namespace trait
    {
        //! The block shared static memory variable allocation operation trait.
        template<typename T, std::size_t TuniqueId, typename TBlockSharedMemSt, typename TSfinae = void>
        struct DeclareSharedVar;
        //! The block shared static memory free operation trait.
        template<typename TBlockSharedMemSt, typename TSfinae = void>
        struct FreeSharedVars;
    } // namespace trait

    //! Declare a block shared variable.
    //!
    //! The variable is uninitialized and not default constructed!
    //! The variable can be accessed by all threads within a block.
    //! Access to the variable is not thread safe.
    //!
    //! \tparam T The element type.
    //! \tparam TuniqueId id those is unique inside a kernel
    //! \tparam TBlockSharedMemSt The block shared allocator implementation type.
    //! \param blockSharedMemSt The block shared allocator implementation.
    //! \return Uninitialized variable stored in shared memory.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, std::size_t TuniqueId, typename TBlockSharedMemSt>
    ALPAKA_FN_ACC auto declareSharedVar(TBlockSharedMemSt const& blockSharedMemSt) -> T&
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptBlockSharedSt, TBlockSharedMemSt>;
        return trait::DeclareSharedVar<T, TuniqueId, ImplementationBase>::declareVar(blockSharedMemSt);
    }

    //! Frees all memory used by block shared variables.
    //!
    //! \tparam TBlockSharedMemSt The block shared allocator implementation type.
    //! \param blockSharedMemSt The block shared allocator implementation.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TBlockSharedMemSt>
    ALPAKA_FN_ACC auto freeSharedVars(TBlockSharedMemSt& blockSharedMemSt) -> void
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptBlockSharedSt, TBlockSharedMemSt>;
        trait::FreeSharedVars<ImplementationBase>::freeVars(blockSharedMemSt);
    }
} // namespace alpaka
