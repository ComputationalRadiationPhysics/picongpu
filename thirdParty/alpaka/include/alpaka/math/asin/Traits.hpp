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
            //! The asin trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Asin;
        }

        //-----------------------------------------------------------------------------
        //! Computes the principal value of the arc sine.
        //!
        //! \tparam TArg The arg type.
        //! \param asin_ctx The object specializing Asin.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto asin(
            T const & asin_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Asin<
                T,
                TArg>
            ::asin(
                asin_ctx,
                arg))
#endif
        {
            return
                traits::Asin<
                    T,
                    TArg>
                ::asin(
                    asin_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Asin specialization for classes with AsinBase member type.
            template<
                typename T,
                typename TArg>
            struct Asin<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::AsinBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto asin(
                    T const & asin_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::asin(
                        static_cast<typename T::AsinBase const &>(asin_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::asin(
                            static_cast<typename T::AsinBase const &>(asin_ctx),
                            arg);
                }
            };
        }
    }
}
