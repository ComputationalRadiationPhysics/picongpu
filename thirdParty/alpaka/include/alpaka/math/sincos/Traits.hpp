/* Copyright 2019 Benjamin Worpitz, Matthias Werner
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
            //! The sincos trait.
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct SinCos;
        }

        //-----------------------------------------------------------------------------
        //! Computes the sine and cosine (measured in radians).
        //!
        //! \tparam T The type of the object specializing SinCos.
        //! \tparam TArg The arg type.
        //! \param sincos_ctx The object specializing SinCos.
        //! \param arg The arg.
        //! \param result_sin result of sine
        //! \param result_cos result of cosine
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto sincos(
            T const & sincos_ctx,
            TArg const & arg,
            TArg & result_sin,
            TArg & result_cos)
        -> void
        {
            traits::SinCos<
                T,
                TArg>
                ::sincos(
                    sincos_ctx,
                    arg,
                    result_sin,
                    result_cos
                    );
        }

        namespace traits
        {
            //#############################################################################
            //! The SinCos specialization for classes with SinCosBase member type.
            template<
                typename T,
                typename TArg>
            struct SinCos<
                T,
                TArg,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename T::SinCosBase,
                        T
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto sincos(
                    T const & sincos_ctx,
                    TArg const & arg,
                    TArg & result_sin,
                    TArg & result_cos
                    )
                -> void
                {
                    // Delegate the call to the base class.
                    math::sincos(
                        static_cast<typename T::SinCosBase const &>(sincos_ctx),
                        arg, result_sin, result_cos);
                }
            };
        }
    }
}
