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
            //! The exp trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Exp;
        }

        //-----------------------------------------------------------------------------
        //! Computes the e (Euler's number, 2.7182818) raised to the given power arg.
        //!
        //! \tparam T The type of the object specializing Exp.
        //! \tparam TArg The arg type.
        //! \param exp_ctx The object specializing Exp.
        //! \param arg The arg.
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto exp(
            T const & exp_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Exp<
                T,
                TArg>
            ::exp(
                exp_ctx,
                arg))
#endif
        {
            return
                traits::Exp<
                    T,
                    TArg>
                ::exp(
                    exp_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Exp specialization for classes with ExpBase member type.
            template<
                typename T,
                typename TArg>
            struct Exp<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::ExpBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto exp(
                    T const & exp_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::exp(
                        static_cast<typename T::ExpBase const &>(exp_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::exp(
                            static_cast<typename T::ExpBase const &>(exp_ctx),
                            arg);
                }
            };
        }
    }
}
