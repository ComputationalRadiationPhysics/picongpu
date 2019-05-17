/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <alpaka/meta/IsStrictBase.hpp>

#include <alpaka/core/Common.hpp>

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
                    return
                        traits::AllocVar<
                            T,
                            TuniqueId,
                            TBlockSharedMemSt>
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
                    traits::FreeMem<
                        TBlockSharedMemSt>
                    ::freeMem(
                        blockSharedMemSt);
                }

                namespace traits
                {
                    //#############################################################################
                    //! The AllocVar trait specialization for classes with BlockSharedMemStBase member type.
                    template<
                        typename T,
                        std::size_t TuniqueId,
                        typename TBlockSharedMemSt>
                    struct AllocVar<
                        T,
                        TuniqueId,
                        TBlockSharedMemSt,
                        typename std::enable_if<
                            meta::IsStrictBase<
                                typename TBlockSharedMemSt::BlockSharedMemStBase,
                                TBlockSharedMemSt
                            >::value
                        >::type>
                    {
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_ACC static auto allocVar(
                            TBlockSharedMemSt const & blockSharedMemSt)
                        -> T &
                        {
                            // Delegate the call to the base class.
                            return
                                block::shared::st::allocVar<
                                    T,
                                    TuniqueId>(
                                        static_cast<typename TBlockSharedMemSt::BlockSharedMemStBase const &>(blockSharedMemSt));
                        }
                    };
                    //#############################################################################
                    //! The FreeMem trait specialization for classes with BlockSharedMemStBase member type.
                    template<
                        typename TBlockSharedMemSt>
                    struct FreeMem<
                        TBlockSharedMemSt,
                        typename std::enable_if<
                            meta::IsStrictBase<
                                typename TBlockSharedMemSt::BlockSharedMemStBase,
                                TBlockSharedMemSt
                            >::value
                        >::type>
                    {
                        //-----------------------------------------------------------------------------
                        ALPAKA_NO_HOST_ACC_WARNING
                        ALPAKA_FN_ACC static auto freeMem(
                            TBlockSharedMemSt & blockSharedMemSt)
                        -> void
                        {
                            // Delegate the call to the base class.
                            block::shared::st::freeMem(
                                static_cast<typename TBlockSharedMemSt::BlockSharedMemStBase &>(blockSharedMemSt));
                        }
                    };
                }
            }
        }
    }
}
