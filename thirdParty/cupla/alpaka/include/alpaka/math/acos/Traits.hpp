/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>

#include <boost/config.hpp>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        struct ConceptMathAcos;

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
                concepts::ImplementationBase<ConceptMathAcos, T>,
                TArg>
            ::acos(
                acos_ctx,
                arg))
#endif
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathAcos, T>;
            return
                traits::Acos<
                    ImplementationBase,
                    TArg>
                ::acos(
                    acos_ctx,
                    arg);
        }
    }
}
