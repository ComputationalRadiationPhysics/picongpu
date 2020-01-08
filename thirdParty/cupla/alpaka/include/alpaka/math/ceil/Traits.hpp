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
            //! The ceil trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Ceil;
        }

        //-----------------------------------------------------------------------------
        //! Computes the smallest integer value not less than arg.
        //!
        //! \tparam T The type of the object specializing Ceil.
        //! \tparam TArg The arg type.
        //! \param ceil_ctx The object specializing Ceil.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto ceil(
            T const & ceil_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Ceil<
                T,
                TArg>
            ::ceil(
                ceil_ctx,
                arg))
#endif
        {
            return
                traits::Ceil<
                    T,
                    TArg>
                ::ceil(
                    ceil_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Ceil specialization for classes with CeilBase member type.
            template<
                typename T,
                typename TArg>
            struct Ceil<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::CeilBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto ceil(
                    T const & ceil_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::ceil(
                        static_cast<typename T::CeilBase const &>(ceil_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::ceil(
                            static_cast<typename T::CeilBase const &>(ceil_ctx),
                            arg);
                }
            };
        }
    }
}
