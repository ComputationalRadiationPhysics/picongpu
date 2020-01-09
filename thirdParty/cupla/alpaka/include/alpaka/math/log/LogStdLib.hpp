/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/math/log/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library log.
        class LogStdLib
        {
        public:
            using LogBase = LogStdLib;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library log trait specialization.
            template<
                typename TArg>
            struct Log<
                LogStdLib,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                ALPAKA_FN_HOST static auto log(
                    LogStdLib const & log_ctx,
                    TArg const & arg)
                -> decltype(std::log(arg))
                {
                    alpaka::ignore_unused(log_ctx);
                    return std::log(arg);
                }
            };
        }
    }
}
