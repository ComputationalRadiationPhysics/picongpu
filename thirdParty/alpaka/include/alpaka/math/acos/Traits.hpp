/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
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
        //! \param acos The object specializing Acos.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto acos(
            T const & acos,
            TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Acos<
                T,
                TArg>
            ::acos(
                acos,
                arg))
#endif
        {
            return
                traits::Acos<
                    T,
                    TArg>
                ::acos(
                    acos,
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
                    T const & acos,
                    TArg const & arg)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    math::acos(
                        static_cast<typename T::AcosBase const &>(acos),
                        arg))
#endif
                {
                    // Delegate the call to the base class.
                    return
                        math::acos(
                            static_cast<typename T::AcosBase const &>(acos),
                            arg);
                }
            };
        }
    }
}
