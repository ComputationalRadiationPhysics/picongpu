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
            //! The log trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Log;
        }

        //-----------------------------------------------------------------------------
        //! Computes the the natural (base e) logarithm of arg.
        //!
        //! \tparam T The type of the object specializing Log.
        //! \tparam TArg The arg type.
        //! \param log_ctx The object specializing Log.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto log(
            T const & log_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Log<
                T,
                TArg>
            ::log(
                log_ctx,
                arg))
#endif
        {
            return
                traits::Log<
                    T,
                    TArg>
                ::log(
                    log_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Log specialization for classes with LogBase member type.
            template<
                typename T,
                typename TArg>
            struct Log<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::LogBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto log(
                    T const & log_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::log(
                        static_cast<typename T::LogBase const &>(log_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::log(
                            static_cast<typename T::LogBase const &>(log_ctx),
                            arg);
                }
            };
        }
    }
}
