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
            //! The atan2 trait.
            template<
                typename T,
                typename Ty,
                typename Tx,
                typename TSfinae = void>
            struct Atan2;
        }

        //-----------------------------------------------------------------------------
        //! Computes the arc tangent of y/x using the signs of arguments to determine the correct quadrant.
        //!
        //! \tparam T The type of the object specializing Atan2.
        //! \tparam Ty The y arg type.
        //! \tparam Tx The x arg type.
        //! \param atan2_ctx The object specializing Atan2.
        //! \param y The y arg.
        //! \param x The x arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename Ty,
            typename Tx>
        ALPAKA_FN_HOST_ACC auto atan2(
            T const & atan2_ctx,
            Ty const & y,
            Tx const & x)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Atan2<
                T,
                Ty,
                Tx>
            ::atan2(
                atan2_ctx,
                y,
                x))
#endif
        {
            return
                traits::Atan2<
                    T,
                    Ty,
                    Tx>
                ::atan2(
                    atan2_ctx,
                    y,
                    x);
        }

        namespace traits
        {
            //#############################################################################
            //! The Atan2 specialization for classes with Atan2Base member type.
            template<
                typename T,
                typename Ty,
                typename Tx>
            struct Atan2<
                T,
                Ty,
                Tx,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::Atan2Base,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto atan2(
                    T const & atan2_ctx,
                    Ty const & y,
                    Tx const & x)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::atan2(
                        static_cast<typename T::Atan2Base const &>(atan2_ctx),
                        y,
                        x))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::atan2(
                            static_cast<typename T::Atan2Base const &>(atan2_ctx),
                            y,
                            x);
                }
            };
        }
    }
}
