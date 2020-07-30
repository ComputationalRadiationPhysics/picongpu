/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera
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
    //-----------------------------------------------------------------------------
    //! The grid block specifics
    namespace block
    {
        //-----------------------------------------------------------------------------
        //! The block shared memory operation specifics.
        namespace shared
        {
            //-----------------------------------------------------------------------------
            //! The block shared static memory operation specifics.
            namespace st
            {
                struct ConceptBlockSharedSt{};

                //-----------------------------------------------------------------------------
                //! The block shared static memory operation traits.
                namespace traits
                {
                    //#############################################################################
                    //! The block shared static memory variable allocation operation trait.
                    template<
                        typename T,
                        std::size_t TuniqueId,
                        typename TBlockSharedMemSt,
                        typename TSfinae = void>
                    struct AllocVar;
                    //#############################################################################
                    //! The block shared static memory free operation trait.
                    template<
                        typename TBlockSharedMemSt,
                        typename TSfinae = void>
                    struct FreeMem;
                }

                //-----------------------------------------------------------------------------
                //! Allocates a variable in block shared static memory.
                //!
                //! The allocated variable is uninitialized and not default constructed!
                //!
                //! \tparam T The element type.
                //! \tparam TuniqueId id those is unique inside a kernel
                //! \tparam TBlockSharedMemSt The block shared allocator implementation type.
                //! \param blockSharedMemSt The block shared allocator implementation.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T,
                    std::size_t TuniqueId,
                    typename TBlockSharedMemSt>
                ALPAKA_FN_ACC auto allocVar(
                    TBlockSharedMemSt const & blockSharedMemSt)
                -> T &
                {
                    using ImplementationBase = concepts::ImplementationBase<ConceptBlockSharedSt, TBlockSharedMemSt>;
                    return
                        traits::AllocVar<
                            T,
                            TuniqueId,
                            ImplementationBase>
                        ::allocVar(
                            blockSharedMemSt);
                }

                //-----------------------------------------------------------------------------
                //! Frees all block shared static memory.
                //!
                //! \tparam TBlockSharedMemSt The block shared allocator implementation type.
                //! \param blockSharedMemSt The block shared allocator implementation.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TBlockSharedMemSt>
                ALPAKA_FN_ACC auto freeMem(
                    TBlockSharedMemSt & blockSharedMemSt)
                -> void
                {
                    using ImplementationBase = concepts::ImplementationBase<ConceptBlockSharedSt, TBlockSharedMemSt>;
                    traits::FreeMem<
                        ImplementationBase>
                    ::freeMem(
                        blockSharedMemSt);
                }
            }
        }
    }
}
