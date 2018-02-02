/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/vec/Vec.hpp>

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
                    ref = ((old == compare) ? value : old);
                    return old;
                }
            };
        }
    }
}
