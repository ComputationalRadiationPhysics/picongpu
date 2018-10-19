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

#include <alpaka/size/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Common.hpp>

#include <boost/config.hpp>

#include <type_traits>
#include <utility>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The work division traits specifics.
    namespace workdiv
    {
        //-----------------------------------------------------------------------------
        //! The work division traits.
        namespace traits
        {
            //#############################################################################
            //! The work div trait.
            template<
                typename TWorkDiv,
                typename TOrigin,
                typename TUnit,
                typename TSfinae = void>
            struct GetWorkDiv;
        }

        //-----------------------------------------------------------------------------
        //! Get the extent requested.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOrigin,
            typename TUnit,
            typename TWorkDiv>
        ALPAKA_FN_HOST_ACC auto getWorkDiv(
            TWorkDiv const & workDiv)
        -> vec::Vec<dim::Dim<TWorkDiv>, size::Size<TWorkDiv>>
        {
            return
                traits::GetWorkDiv<
                    TWorkDiv,
                    TOrigin,
                    TUnit>
                ::getWorkDiv(
                    workDiv);
        }

        namespace traits
        {
            //#############################################################################
            //! The work div grid block extent trait specialization for classes with WorkDivBase member type.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Grid,
                unit::Blocks,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename TWorkDiv::WorkDivBase,
                        TWorkDiv
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
                -> vec::Vec<dim::Dim<typename TWorkDiv::WorkDivBase>, size::Size<typename TWorkDiv::WorkDivBase>>
                {
                    // Delegate the call to the base class.
                    return
                        workdiv::getWorkDiv<
                            origin::Grid,
                            unit::Blocks>(
                                static_cast<typename TWorkDiv::WorkDivBase const &>(workDiv));
                }
            };
            //#############################################################################
            //! The work div block thread extent trait specialization for classes with WorkDivBase member type.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Block,
                unit::Threads,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename TWorkDiv::WorkDivBase,
                        TWorkDiv
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
                -> vec::Vec<dim::Dim<typename TWorkDiv::WorkDivBase>, size::Size<typename TWorkDiv::WorkDivBase>>
                {
                    // Delegate the call to the base class.
                    return
                        workdiv::getWorkDiv<
                            origin::Block,
                            unit::Threads>(
                                static_cast<typename TWorkDiv::WorkDivBase const &>(workDiv));
                }
            };
            //#############################################################################
            //! The work div block thread extent trait specialization for classes with WorkDivBase member type.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Thread,
                unit::Elems,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename TWorkDiv::WorkDivBase,
                        TWorkDiv
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
                -> vec::Vec<dim::Dim<typename TWorkDiv::WorkDivBase>, size::Size<typename TWorkDiv::WorkDivBase>>
                {
                    // Delegate the call to the base class.
                    return
                        workdiv::getWorkDiv<
                            origin::Thread,
                            unit::Elems>(
                                static_cast<typename TWorkDiv::WorkDivBase const &>(workDiv));
                }
            };

            //#############################################################################
            //! The work div grid thread extent trait specialization.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Grid,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    workdiv::getWorkDiv<origin::Grid, unit::Blocks>(workDiv)
                    * workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv))
#endif
                {
                    return
                        workdiv::getWorkDiv<origin::Grid, unit::Blocks>(workDiv)
                        * workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv);
                }
            };
            //#############################################################################
            //! The work div grid element extent trait specialization.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Grid,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    workdiv::getWorkDiv<origin::Grid, unit::Threads>(workDiv)
                    * workdiv::getWorkDiv<origin::Thread, unit::Elems>(workDiv))
#endif
                {
                    return
                        workdiv::getWorkDiv<origin::Grid, unit::Threads>(workDiv)
                        * workdiv::getWorkDiv<origin::Thread, unit::Elems>(workDiv);
                }
            };
            //#############################################################################
            //! The work div block element extent trait specialization.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Block,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv)
                    * workdiv::getWorkDiv<origin::Thread, unit::Elems>(workDiv))
#endif
                {
                    return
                        workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv)
                        * workdiv::getWorkDiv<origin::Thread, unit::Elems>(workDiv);
                }
            };
        }
    }
}
