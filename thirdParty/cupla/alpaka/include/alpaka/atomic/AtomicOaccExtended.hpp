/* Copyright 2022 Jeffrey Kelling, Bernhard Manfred Gruber
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#    if _OPENACC < 201306
#        error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC 2.0 or higher!
#    endif

#    include <alpaka/atomic/AtomicOaccBuiltIn.hpp>
#    include <alpaka/atomic/Traits.hpp>
#    include <alpaka/core/Positioning.hpp>

namespace alpaka
{
    template<class THierarchy>
    struct AtomicOaccExtended : public AtomicOaccBuiltIn
    {
        AtomicOaccExtended(std::uint32_t* pmutex) : mutex(pmutex)
        {
        }

        std::uint32_t* const mutex;
    };

    template<>
    struct AtomicOaccExtended<hierarchy::Threads> : public AtomicOaccBuiltIn
    {
        mutable std::uint32_t mutex[2] = {0u, 0u};
    };

    namespace trait
    {
        namespace detail
        {
            template<typename Op, typename THierarchy>
            auto criticalOp(Op op, AtomicOaccExtended<THierarchy> const& atomic)
            {
                std::uint32_t ticket;
#    pragma acc atomic capture
                {
                    ticket = atomic.mutex[0];
                    atomic.mutex[0] += 1;
                }
                std::uint32_t queue;
                do
                {
#    pragma acc atomic read
                    queue = atomic.mutex[1];
                } while(queue != ticket);

                auto ret = op();

#    pragma acc atomic update
                atomic.mutex[1] += 1;

                return ret;
            }
        } // namespace detail

        //! Forward traits for OpenACC built-in atomic operations
        template<typename TOp, typename T, typename THierarchy>
        struct AtomicOp<TOp, AtomicOaccExtended<THierarchy>, T, THierarchy>
            : public AtomicOp<TOp, AtomicOaccBuiltIn, T, THierarchy>
        {
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccExtended<THierarchy> const& atomic,
                T* const addr,
                T const& value) -> T
            {
                return AtomicOp<TOp, AtomicOaccBuiltIn, T, THierarchy>::atomicOp(atomic, addr, value);
            }
        };

        //! The OpenACC accelerators atomic operation: Min
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicMin, AtomicOaccExtended<THierarchy>, T, THierarchy>
        {
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccExtended<THierarchy> const& atomic,
                T* const addr,
                T const& value) -> T
            {
                return detail::criticalOp(
                    [&
#    if BOOST_COMP_PGI
                     ,
                     addr // NVHPC 21.7: capturing pointer by ref results in invalid address inside lambda
#    endif
                ]()
                    {
                        auto& ref(*addr);
                        T old = ref;
                        ref = (ref <= value) ? ref : value;
                        return old;
                    },
                    atomic);
            }
        };

        //#############################################################################
        //! The OpenACC accelerators atomic operation: Max
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicMax, AtomicOaccExtended<THierarchy>, T, THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccExtended<THierarchy> const& atomic,
                T* const addr,
                T const& value) -> T
            {
                return detail::criticalOp(
                    [&
#    if BOOST_COMP_PGI
                     ,
                     addr // NVHPC 21.7: capturing pointer by ref results in invalid address inside lambda
#    endif
                ]()
                    {
                        auto& ref(*addr);
                        T old = ref;
                        ref = (ref >= value) ? ref : value;
                        return old;
                    },
                    atomic);
            }
        };

        //! The OpenACC accelerators atomic operation: Inc
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicInc, AtomicOaccExtended<THierarchy>, T, THierarchy>
        {
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccExtended<THierarchy> const& atomic,
                T* const addr,
                T const& value) -> T
            {
                return detail::criticalOp(
                    [&
#    if BOOST_COMP_PGI
                     ,
                     addr // NVHPC 21.7: capturing pointer by ref results in invalid address inside lambda
#    endif
                ]()
                    {
                        auto& ref(*addr);
                        T old = ref;
                        ref = ((ref >= value) ? 0 : (ref + 1));
                        return old;
                    },
                    atomic);
            }
        };

        //! The OpenACC accelerators atomic operation: Dec
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicDec, AtomicOaccExtended<THierarchy>, T, THierarchy>
        {
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccExtended<THierarchy> const& atomic,
                T* const addr,
                T const& value) -> T
            {
                return detail::criticalOp(
                    [&
#    if BOOST_COMP_PGI
                     ,
                     addr // NVHPC 21.7: capturing pointer by ref results in invalid address inside lambda
#    endif
                ]()
                    {
                        auto& ref(*addr);
                        T old = ref;
                        ref = ((ref == 0) || (ref > value)) ? value : (ref - 1);
                        return old;
                    },
                    atomic);
            }
        };

        //! The OpenACC accelerators atomic operation: Cas
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicCas, AtomicOaccExtended<THierarchy>, T, THierarchy>
        {
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccExtended<THierarchy> const& atomic,
                T* const addr,
                T const& compare,
                T const& value) -> T
            {
                return detail::criticalOp(
                    [&
#    if BOOST_COMP_PGI
                     ,
                     addr // NVHPC 21.7: capturing pointer by ref results in invalid address inside lambda
#    endif
                ]()
                    {
                        auto& ref(*addr);
                        T old = ref;
#    if defined(__GNUC__)
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wfloat-equal"
#    endif
                        ref = (ref == compare ? value : ref);
#    if defined(__GNUC__)
#        pragma GCC diagnostic pop
#    endif
                        return old;
                    },
                    atomic);
            }
        };

    } // namespace trait
} // namespace alpaka

#endif
