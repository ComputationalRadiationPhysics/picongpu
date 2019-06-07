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
            //! The rsqrt trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Rsqrt;
        }

        //-----------------------------------------------------------------------------
        //! Computes the rsqrt.
        //!
        //! \tparam T The type of the object specializing Rsqrt.
        //! \tparam TArg The arg type.
        //! \param rsqrt_ctx The object specializing Rsqrt.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto rsqrt(
            T const & rsqrt_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Rsqrt<
                T,
                TArg>
            ::rsqrt(
                rsqrt_ctx,
                arg))
#endif
        {
            return
                traits::Rsqrt<
                    T,
                    TArg>
                ::rsqrt(
                    rsqrt_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Rsqrt specialization for classes with RsqrtBase member type.
            template<
                typename T,
                typename TArg>
            struct Rsqrt<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::RsqrtBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto rsqrt(
                    T const & rsqrt_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::rsqrt(
                        static_cast<typename T::RsqrtBase const &>(rsqrt_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::rsqrt(
                            static_cast<typename T::RsqrtBase const &>(rsqrt_ctx),
                            arg);
                }
            };
        }
    }
}
