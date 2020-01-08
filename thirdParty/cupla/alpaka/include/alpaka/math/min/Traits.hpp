/* Copyright 2019 Benjamin Worpitz
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

#include <boost/config.hpp>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            //#############################################################################
            //! The min trait.
            template<
                typename T,
                typename Tx,
                typename Ty,
                typename TSfinae = void>
            struct Min;
        }

        //-----------------------------------------------------------------------------
        //! Returns the smaller of two arguments.
        //! NaNs are treated as missing data (between a NaN and a numeric value, the numeric value is chosen).
        //!
        //! \tparam T The type of the object specializing Min.
        //! \tparam Tx The type of the first argument.
        //! \tparam Ty The type of the second argument.
        //! \param min_ctx The object specializing Min.
        //! \param x The first argument.
        //! \param y The second argument.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename Tx,
            typename Ty>
        ALPAKA_FN_HOST_ACC auto min(
            T const & min_ctx,
            Tx const & x,
            Ty const & y)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Min<
                T,
                Tx,
                Ty>
            ::min(
                min_ctx,
                x,
                y))
#endif
        {
            return
                traits::Min<
                    T,
                    Tx,
                    Ty>
                ::min(
                    min_ctx,
                    x,
                    y);
        }

        namespace traits
        {
            //#############################################################################
            //! The Min specialization for classes with MinBase member type.
            template<
                typename T,
                typename Tx,
                typename Ty>
            struct Min<
                T,
                Tx,
                Ty,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::MinBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto min(
                    T const & min_ctx,
                    Tx const & x,
                    Ty const & y)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::min(
                        static_cast<typename T::MinBase const &>(min_ctx),
                        x,
                        y))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::min(
                            static_cast<typename T::MinBase const &>(min_ctx),
                            x,
                            y);
                }
            };
        }
    }
}
