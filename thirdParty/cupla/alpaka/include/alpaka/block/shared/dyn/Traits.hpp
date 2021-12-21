/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>

#include <type_traits>

namespace alpaka
{
    struct ConceptBlockSharedDyn
    {
    };

    //-----------------------------------------------------------------------------
    //! The block shared dynamic memory operation traits.
    namespace traits
    {
        //#############################################################################
        //! The block shared dynamic memory get trait.
        template<typename T, typename TBlockSharedMemDyn, typename TSfinae = void>
        struct GetDynSharedMem;
    } // namespace traits

    //-----------------------------------------------------------------------------
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
        return traits::GetDynSharedMem<T, ImplementationBase>::getMem(blockSharedMemDyn);
    }
} // namespace alpaka
