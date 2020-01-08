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
            //! The trunc trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Trunc;
        }

        //-----------------------------------------------------------------------------
        //! Computes the nearest integer not greater in magnitude than arg.
        //!
        //! \tparam T The type of the object specializing Trunc.
        //! \tparam TArg The arg type.
        //! \param trunc_ctx The object specializing Trunc.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto trunc(
            T const & trunc_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Trunc<
                T,
                TArg>
            ::trunc(
                trunc_ctx,
                arg))
#endif
        {
            return
                traits::Trunc<
                    T,
                    TArg>
                ::trunc(
                    trunc_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Trunc specialization for classes with TruncBase member type.
            template<
                typename T,
                typename TArg>
            struct Trunc<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::TruncBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto trunc(
                    T const & trunc_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::trunc(
                        static_cast<typename T::TruncBase const &>(trunc_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::trunc(
                            static_cast<typename T::TruncBase const &>(trunc_ctx),
                            arg);
                }
            };
        }
    }
}
