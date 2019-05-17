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
            //! The cbrt trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Cbrt;
        }

        //-----------------------------------------------------------------------------
        //! Computes the cbrt.
        //!
        //! \tparam T The type of the object specializing Cbrt.
        //! \tparam TArg The arg type.
        //! \param cbrt_ctx The object specializing Cbrt.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto cbrt(
            T const & cbrt_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Cbrt<
                T,
                TArg>
            ::cbrt(
                cbrt_ctx,
                arg))
#endif
        {
            return
                traits::Cbrt<
                    T,
                    TArg>
                ::cbrt(
                    cbrt_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Cbrt specialization for classes with CbrtBase member type.
            template<
                typename T,
                typename TArg>
            struct Cbrt<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::CbrtBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto cbrt(
                    T const & cbrt_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::cbrt(
                        static_cast<typename T::CbrtBase const &>(cbrt_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::cbrt(
                            static_cast<typename T::CbrtBase const &>(cbrt_ctx),
                            arg);
                }
            };
        }
    }
}
