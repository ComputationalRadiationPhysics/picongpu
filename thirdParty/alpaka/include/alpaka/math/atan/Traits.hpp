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
            //! The atan trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Atan;
        }

        //-----------------------------------------------------------------------------
        //! Computes the principal value of the arc tangent.
        //!
        //! \tparam TArg The arg type.
        //! \param atan_ctx The object specializing Atan.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto atan(
            T const & atan_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Atan<
                T,
                TArg>
            ::atan(
                atan_ctx,
                arg))
#endif
        {
            return
                traits::Atan<
                    T,
                    TArg>
                ::atan(
                    atan_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Atan specialization for classes with AtanBase member type.
            template<
                typename T,
                typename TArg>
            struct Atan<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::AtanBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto atan(
                    T const & atan_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::atan(
                        static_cast<typename T::AtanBase const &>(atan_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::atan(
                            static_cast<typename T::AtanBase const &>(atan_ctx),
                            arg);
                }
            };
        }
    }
}
