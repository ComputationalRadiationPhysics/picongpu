/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Common.hpp>

#include <algorithm>

namespace alpaka
{
    namespace atomic
    {
        //-----------------------------------------------------------------------------
        //! Defines operation functors.
        namespace op
        {
            //#############################################################################
            //! The addition function object.
            struct Add
            {
                //-----------------------------------------------------------------------------
                //! \return The old value of addr.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T>
                ALPAKA_FN_HOST_ACC auto operator()(
                    T * const addr,
                    T const & value) const
                -> T
                {
                    auto const old(*addr);
                    auto & ref(*addr);
                    ref += value;
                    return old;
                }
            };
            //#############################################################################
            //! The subtraction function object.
            struct Sub
            {
                //-----------------------------------------------------------------------------
                //! \return The old value of addr.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T>
                ALPAKA_FN_HOST_ACC auto operator()(
                    T * const addr,
                    T const & value) const
                -> T
                {
                    auto const old(*addr);
                    auto & ref(*addr);
                    ref -= value;
                    return old;
                }
            };
            //#############################################################################
            //! The minimum function object.
            struct Min
            {
                //-----------------------------------------------------------------------------
                //! \return The old value of addr.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T>
                ALPAKA_FN_HOST_ACC auto operator()(
                    T * const addr,
                    T const & value) const
                -> T
                {
                    auto const old(*addr);
                    auto & ref(*addr);
                    ref = std::min(ref, value);
                    return old;
                }
            };
            //#############################################################################
            //! The maximum function object.
            struct Max
            {
                //-----------------------------------------------------------------------------
                //! \return The old value of addr.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T>
                ALPAKA_FN_HOST_ACC auto operator()(
                    T * const addr,
                    T const & value) const
                -> T
                {
                    auto const old(*addr);
                    auto & ref(*addr);
                    ref = std::max(ref, value);
                    return old;
                }
            };
            //#############################################################################
            //! The exchange function object.
            struct Exch
            {
                //-----------------------------------------------------------------------------
                //! \return The old value of addr.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T>
                ALPAKA_FN_HOST_ACC auto operator()(
                    T * const addr,
                    T const & value) const
                -> T
                {
                    auto const old(*addr);
                    auto & ref(*addr);
                    ref = value;
                    return old;
                }
            };
            //#############################################################################
            //! The increment function object.
            struct Inc
            {
                //-----------------------------------------------------------------------------
                //! Increments up to value, then reset to 0.
                //!
                //! \return The old value of addr.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T>
                ALPAKA_FN_HOST_ACC auto operator()(
                    T * const addr,
                    T const & value) const
                -> T
                {
                    auto const old(*addr);
                    auto & ref(*addr);
                    ref = ((old >= value) ? 0 : old + 1);
                    return old;
                }
            };
            //#############################################################################
            //! The decrement function object.
            struct Dec
            {
                //-----------------------------------------------------------------------------
                //! Decrement down to 0, then reset to value.
                //!
                //! \return The old value of addr.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T>
                ALPAKA_FN_HOST_ACC auto operator()(
                    T * const addr,
                    T const & value) const
                -> T
                {
                    auto const old(*addr);
                    auto & ref(*addr);
                    ref = (((old == 0) || (old > value)) ? value : (old - 1));
                    return old;
                }
            };
            //#############################################################################
            //! The and function object.
            struct And
            {
                //-----------------------------------------------------------------------------
                //! \return The old value of addr.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T>
                ALPAKA_FN_HOST_ACC auto operator()(
                    T * const addr,
                    T const & value) const
                -> T
                {
                    auto const old(*addr);
                    auto & ref(*addr);
                    ref &= value;
                    return old;
                }
            };
            //#############################################################################
            //! The or function object.
            struct Or
            {
                //-----------------------------------------------------------------------------
                //! \return The old value of addr.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T>
                ALPAKA_FN_HOST_ACC auto operator()(
                    T * const addr,
                    T const & value) const
                -> T
                {
                    auto const old(*addr);
                    auto & ref(*addr);
                    ref |= value;
                    return old;
                }
            };
            //#############################################################################
            //! The exclusive or function object.
            struct Xor
            {
                //-----------------------------------------------------------------------------
                //! \return The old value of addr.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T>
                ALPAKA_FN_HOST_ACC auto operator()(
                    T * const addr,
                    T const & value) const
                -> T
                {
                    auto const old(*addr);
                    auto & ref(*addr);
                    ref ^= value;
                    return old;
                }
            };
            //#############################################################################
            //! The compare and swap function object.
            struct Cas
            {
                //-----------------------------------------------------------------------------
                //! \return The old value of addr.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T>
                ALPAKA_FN_HOST_ACC auto operator()(
                    T * addr,
                    T const & compare,
                    T const & value) const
                -> T
                {
                    auto const old(*addr);
                    auto & ref(*addr);

// gcc-7.4.0 assumes for an optimization that a signed overflow does not occur here.
// That's fine, so ignore that warning.
#if BOOST_COMP_GNUC && (BOOST_COMP_GNUC == BOOST_VERSION_NUMBER(7, 4, 0))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#endif
                    ref = ((old == compare) ? value : old);
#if BOOST_COMP_GNUC && (BOOST_COMP_GNUC == BOOST_VERSION_NUMBER(7, 4, 0))
#pragma GCC diagnostic pop
#endif
                    return old;
                }
            };
        }
    }
}
