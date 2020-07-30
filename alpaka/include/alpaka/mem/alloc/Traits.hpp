/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/extent/Traits.hpp>

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>

namespace alpaka
{
    namespace mem
    {
        //-----------------------------------------------------------------------------
        //! The allocator specifics.
        namespace alloc
        {
            struct ConceptMemAlloc{};

            //-----------------------------------------------------------------------------
            //! The allocator traits.
            namespace traits
            {
                //#############################################################################
                //! The memory allocation trait.
                template<
                    typename T,
                    typename TAlloc,
                    typename TSfinae = void>
                struct Alloc;

                //#############################################################################
                //! The memory free trait.
                template<
                    typename T,
                    typename TAlloc,
                    typename TSfinae = void>
                struct Free;
            }

            //-----------------------------------------------------------------------------
            //! \return The pointer to the allocated memory.
            template<
                typename T,
                typename TAlloc>
            ALPAKA_FN_HOST auto alloc(
                TAlloc const & alloc,
                std::size_t const & sizeElems)
            -> T *
            {
                using ImplementationBase = concepts::ImplementationBase<ConceptMemAlloc, TAlloc>;
                return
                    traits::Alloc<
                        T,
                        ImplementationBase>
                    ::alloc(
                        alloc,
                        sizeElems);
            }

            //-----------------------------------------------------------------------------
            //! Frees the memory identified by the given pointer.
            template<
                typename TAlloc,
                typename T>
            ALPAKA_FN_HOST auto free(
                TAlloc const & alloc,
                T const * const ptr)
            -> void
            {
                using ImplementationBase = concepts::ImplementationBase<ConceptMemAlloc, TAlloc>;
                traits::Free<
                    T,
                    ImplementationBase>
                ::free(
                    alloc,
                    ptr);
            }
        }
    }
}
