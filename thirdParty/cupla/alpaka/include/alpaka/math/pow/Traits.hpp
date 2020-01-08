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
            //! The pow trait.
            template<
                typename T,
                typename TBase,
                typename TExp,
                typename TSfinae = void>
            struct Pow;
        }

        //-----------------------------------------------------------------------------
        //! Computes the value of base raised to the power exp.
        //!
        //! \tparam T The type of the object specializing Pow.
        //! \tparam TBase The base type.
        //! \tparam TExp The exponent type.
        //! \param pow_ctx The object specializing Pow.
        //! \param base The base.
        //! \param exp The exponent.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TBase,
            typename TExp>
        ALPAKA_FN_HOST_ACC auto pow(
            T const & pow_ctx,
            TBase const & base,
            TExp const & exp)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Pow<
                T,
                TBase,
                TExp>
            ::pow(
                pow_ctx,
                base,
                exp))
#endif
        {
            return
                traits::Pow<
                    T,
                    TBase,
                    TExp>
                ::pow(
                    pow_ctx,
                    base,
                    exp);
        }

        namespace traits
        {
            //#############################################################################
            //! The Pow specialization for classes with PowBase member type.
            template<
                typename T,
                typename TBase,
                typename TExp>
            struct Pow<
                T,
                TBase,
                TExp,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::PowBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto pow(
                    T const & pow_ctx,
                    TBase const & base,
                    TExp const & exp)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::pow(
                        static_cast<typename T::PowBase const &>(pow_ctx),
                        base,
                        exp))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::pow(
                            static_cast<typename T::PowBase const &>(pow_ctx),
                            base,
                            exp);
                }
            };
        }
    }
}
