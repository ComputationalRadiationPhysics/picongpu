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
            //! The sqrt trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Sqrt;
        }

        //-----------------------------------------------------------------------------
        //! Computes the square root of arg.
        //!
        //! \tparam T The type of the object specializing Sqrt.
        //! \tparam TArg The arg type.
        //! \param sqrt_ctx The object specializing Sqrt.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto sqrt(
            T const & sqrt_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Sqrt<
                T,
                TArg>
            ::sqrt(
                sqrt_ctx,
                arg))
#endif
        {
            return
                traits::Sqrt<
                    T,
                    TArg>
                ::sqrt(
                    sqrt_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Sqrt specialization for classes with SqrtBase member type.
            template<
                typename T,
                typename TArg>
            struct Sqrt<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::SqrtBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto sqrt(
                    T const & sqrt_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::sqrt(
                        static_cast<typename T::SqrtBase const &>(sqrt_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::sqrt(
                            static_cast<typename T::SqrtBase const &>(sqrt_ctx),
                            arg);
                }
            };
        }
    }
}
