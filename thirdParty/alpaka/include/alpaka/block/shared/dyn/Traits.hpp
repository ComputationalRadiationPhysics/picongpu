/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"

#include <type_traits>

namespace alpaka
{
    struct ConceptBlockSharedDyn
    {
    };

    //! The block shared dynamic memory operation traits.
    namespace trait
    {
        //! The block shared dynamic memory get trait.
        template<typename T, typename TBlockSharedMemDyn, typename TSfinae = void>
        struct GetDynSharedMem;
    } // namespace trait

    //! Get block shared dynamic memory.
    //!
    //! The available size of the memory can be defined by specializing the trait
    //! BlockSharedMemDynSizeBytes for a kernel.
    //! The Memory can be accessed by all threads within a block.
    //! Access to the memory is not thread safe.
    //!
    //! \tparam T The element type.
    //! \tparam TBlockSharedMemDyn The block shared dynamic memory implementation type.
    //! \param blockSharedMemDyn The block shared dynamic memory implementation.
    //! \return Pointer to pre-allocated contiguous memory.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TBlockSharedMemDyn>
    ALPAKA_FN_ACC auto getDynSharedMem(TBlockSharedMemDyn const& blockSharedMemDyn) -> T*
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptBlockSharedDyn, TBlockSharedMemDyn>;
        return trait::GetDynSharedMem<T, ImplementationBase>::getMem(blockSharedMemDyn);
    }
} // namespace alpaka
