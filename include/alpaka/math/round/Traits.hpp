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
#include <alpaka/core/Unused.hpp>

#include <boost/config.hpp>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            //#############################################################################
            //! The round trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Round;

            //#############################################################################
            //! The round trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Lround;

            //#############################################################################
            //! The round trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Llround;
        }

        //-----------------------------------------------------------------------------
        //! Computes the nearest integer value to arg (in floating-point format), rounding halfway cases away from zero, regardless of the current rounding mode.
        //!
        //! \tparam T The type of the object specializing Round.
        //! \tparam TArg The arg type.
        //! \param round_ctx The object specializing Round.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto round(
            T const & round_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Round<
                T,
                TArg>
            ::round(
                round_ctx,
                arg))
#endif
        {
            return
                traits::Round<
                    T,
                    TArg>
                ::round(
                    round_ctx,
                    arg);
        }
        //-----------------------------------------------------------------------------
        //! Computes the nearest integer value to arg (in integer format), rounding halfway cases away from zero, regardless of the current rounding mode.
        //!
        //! \tparam T The type of the object specializing Round.
        //! \tparam TArg The arg type.
        //! \param lround_ctx The object specializing Round.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto lround(
            T const & lround_ctx,
            TArg const & arg)
        -> long int
        {
            return
                traits::Lround<
                    T,
                    TArg>
                ::lround(
                    lround_ctx,
                    arg);
        }
        //-----------------------------------------------------------------------------
        //! Computes the nearest integer value to arg (in integer format), rounding halfway cases away from zero, regardless of the current rounding mode.
        //!
        //! \tparam T The type of the object specializing Round.
        //! \tparam TArg The arg type.
        //! \param llround_ctx The object specializing Round.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto llround(
            T const & llround_ctx,
            TArg const & arg)
        -> long long int
        {
            return
                traits::Llround<
                    T,
                    TArg>
                ::llround(
                    llround_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Round specialization for classes with RoundBase member type.
            template<
                typename T,
                typename TArg>
            struct Round<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::RoundBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto round(
                    T const & round_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::round(
                        static_cast<typename T::RoundBase const &>(round_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::round(
                            static_cast<typename T::RoundBase const &>(round_ctx),
                            arg);
                }
            };
            //#############################################################################
            //! The Lround specialization for classes with RoundBase member type.
            template<
                typename T,
                typename TArg>
            struct Lround<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::RoundBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto lround(
                    T const & lround_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::lround(
                        static_cast<typename T::RoundBase const &>(lround_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::lround(
                            static_cast<typename T::RoundBase const &>(lround_ctx),
                            arg);
                }
            };
            //#############################################################################
            //! The Llround specialization for classes with RoundBase member type.
            template<
                typename T,
                typename TArg>
            struct Llround<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::RoundBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto llround(
                    T const & llround_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::llround(
                        static_cast<typename T::RoundBase const &>(llround_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::llround(
                            static_cast<typename T::RoundBase const &>(llround_ctx),
                            arg);
                }
            };
        }
    }
}
