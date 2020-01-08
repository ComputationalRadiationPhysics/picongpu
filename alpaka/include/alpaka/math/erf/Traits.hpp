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
            //! The erf trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Erf;
        }

        //-----------------------------------------------------------------------------
        //! Computes the error function of arg.
        //!
        //! \tparam T The type of the object specializing Erf.
        //! \tparam TArg The arg type.
        //! \param erf_ctx The object specializing Erf.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto erf(
            T const & erf_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Erf<
                T,
                TArg>
            ::erf(
                erf_ctx,
                arg))
#endif
        {
            return
                traits::Erf<
                    T,
                    TArg>
                ::erf(
                    erf_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Erf specialization for classes with ErfBase member type.
            template<
                typename T,
                typename TArg>
            struct Erf<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::ErfBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto erf(
                    T const & erf_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::erf(
                        static_cast<typename T::ErfBase const &>(erf_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::erf(
                            static_cast<typename T::ErfBase const &>(erf_ctx),
                            arg);
                }
            };
        }
    }
}
