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
            //! The sin trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Sin;
        }

        //-----------------------------------------------------------------------------
        //! Computes the sine (measured in radians).
        //!
        //! \tparam T The type of the object specializing Sin.
        //! \tparam TArg The arg type.
        //! \param sin_ctx The object specializing Sin.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto sin(
            T const & sin_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Sin<
                T,
                TArg>
            ::sin(
                sin_ctx,
                arg))
#endif
        {
            return
                traits::Sin<
                    T,
                    TArg>
                ::sin(
                    sin_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Sin specialization for classes with SinBase member type.
            template<
                typename T,
                typename TArg>
            struct Sin<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::SinBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto sin(
                    T const & sin_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::sin(
                        static_cast<typename T::SinBase const &>(sin_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::sin(
                            static_cast<typename T::SinBase const &>(sin_ctx),
                            arg);
                }
            };
        }
    }
}
