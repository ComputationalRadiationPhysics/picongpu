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
        //-----------------------------------------------------------------------------
        //! The math traits.
        namespace traits
        {
            //#############################################################################
            //! The abs trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Abs;
        }

        //-----------------------------------------------------------------------------
        //! Computes the absolute value.
        //!
        //! \tparam T The type of the object specializing Abs.
        //! \tparam TArg The arg type.
        //! \param abs_ctx The object specializing Abs.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto abs(
            T const & abs_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Abs<
                T,
                TArg>
            ::abs(
                abs_ctx,
                arg))
#endif
        {
            return
                traits::Abs<
                    T,
                    TArg>
                ::abs(
                    abs_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Abs specialization for classes with AbsBase member type.
            template<
                typename T,
                typename TArg>
            struct Abs<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::AbsBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto abs(
                    T const & abs_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::abs(
                        static_cast<typename T::AbsBase const &>(abs_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::abs(
                            static_cast<typename T::AbsBase const &>(abs_ctx),
                            arg);
                }
            };
        }
    }
}
