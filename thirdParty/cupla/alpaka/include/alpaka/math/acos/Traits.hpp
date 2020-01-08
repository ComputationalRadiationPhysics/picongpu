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
            //! The acos trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Acos;
        }

        //-----------------------------------------------------------------------------
        //! Computes the principal value of the arc cosine.
        //!
        //! \tparam TArg The arg type.
        //! \param acos_ctx The object specializing Acos.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto acos(
            T const & acos_ctx,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Acos<
                T,
                TArg>
            ::acos(
                acos_ctx,
                arg))
#endif
        {
            return
                traits::Acos<
                    T,
                    TArg>
                ::acos(
                    acos_ctx,
                    arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Acos specialization for classes with AcosBase member type.
            template<
                typename T,
                typename TArg>
            struct Acos<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::AcosBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto acos(
                    T const & acos_ctx,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::acos(
                        static_cast<typename T::AcosBase const &>(acos_ctx),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::acos(
                            static_cast<typename T::AcosBase const &>(acos_ctx),
                            arg);
                }
            };
        }
    }
}
