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
            //! The tan trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Tan;
        }

        //-----------------------------------------------------------------------------
        //! Computes the tangent (measured in radians).
        //!
        //! \tparam T The type of the object specializing Tan.
        //! \tparam TArg The arg type.
        //! \param tan_ctx The object specializing Tan.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto tan(
            T const & tan_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Tan<
                T,
                TArg>
            ::tan(
                tan_ctx,
                arg))
#endif
        {
            return
                traits::Tan<
                    T,
                    TArg>
                ::tan(
                    tan_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Tan specialization for classes with TanBase member type.
            template<
                typename T,
                typename TArg>
            struct Tan<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::TanBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                //
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto tan(
                    T const & tan_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::tan(
                        static_cast<typename T::TanBase const &>(tan_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::tan(
                            static_cast<typename T::TanBase const &>(tan_ctx),
                            arg);
                }
            };
        }
    }
}
