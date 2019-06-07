/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
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

namespace alpaka
{
    namespace mem
    {
        //-----------------------------------------------------------------------------
        //! The allocator specifics.
        namespace alloc
        {
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
                return
                    traits::Alloc<
                        T,
                        TAlloc>
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
                traits::Free<
                    T,
                    TAlloc>
                ::free(
                    alloc,
                    ptr);
            }

            namespace traits
            {
                //#############################################################################
                //! The Alloc specialization for classes with AllocBase member type.
                template<
                    typename T,
                    typename TAlloc>
                struct Alloc<
                    T,
                    TAlloc,
                    typename std::enable_if<
                        std::is_base_of<typename TAlloc::AllocBase, typename std::decay<TAlloc>::type>::value
                        && (!std::is_same<typename TAlloc::AllocBase, typename std::decay<TAlloc>::type>::value)>::type>
                {
                    //-----------------------------------------------------------------------------
                    // FIXME: compiler switch for the host-device signatures
#if defined( BOOST_COMP_HCC ) && BOOST_COMP_HCC
                    ALPAKA_FN_HOST
#else
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC
#endif
                    static auto alloc(
                        TAlloc const & alloc,
                        std::size_t const & sizeElems)
                    -> T *
                    {
                        // Delegate the call to the base class.
                        return
                            mem::alloc::alloc<
                                T>(
                                    static_cast<typename TAlloc::AllocBase const &>(alloc),
                                    sizeElems);
                    }
                };

                //#############################################################################
                //! The Free specialization for classes with AllocBase member type.
                template<
                    typename T,
                    typename TAlloc>
                struct Free<
                    T,
                    TAlloc,
                    typename std::enable_if<
                        std::is_base_of<typename TAlloc::AllocBase, typename std::decay<TAlloc>::type>::value
                        && (!std::is_same<typename TAlloc::AllocBase, typename std::decay<TAlloc>::type>::value)>::type>
                {
                    //-----------------------------------------------------------------------------
                    // FIXME: compiler switch for the host-device signatures
#if defined( BOOST_COMP_HCC ) && BOOST_COMP_HCC
                    ALPAKA_FN_HOST
#else
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC
#endif
                    static auto free(
                        TAlloc const & alloc,
                        T const * const ptr)
                    -> void
                    {
                        // Delegate the call to the base class.
                        mem::alloc::free(
                            static_cast<typename TAlloc::AllocBase const &>(alloc),
                            ptr);
                    }
                };
            }
        }
    }
}
