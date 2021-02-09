/* Copyright 2013-2021 Heiko Burau, Rene Widera, Axel Huebl
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"
#include "pmacc/algorithms/TypeCast.hpp"
#include "pmacc/math/vector/Int.hpp"
#include "pmacc/math/Vector.hpp"
#include "pmacc/math/VectorOperations.hpp"

namespace pmacc
{
    namespace algorithm
    {
        namespace cuplaBlock
        {
#ifndef FOREACH_KERNEL_MAX_PARAMS
#    define FOREACH_KERNEL_MAX_PARAMS 4
#endif

#define SHIFTACCESS_CURSOR(Z, N, _) c##N[pos]

#define FOREACH_OPERATOR(Z, N, _)                                                                                     \
    /*      <             , typename C0, ..., typename C(N-1)  ,              > */                                    \
    template<                                                                                                         \
        typename Zone,                                                                                                \
        BOOST_PP_ENUM_PARAMS(N, typename C),                                                                          \
        typename Functor,                                                                                             \
        typename T_Acc> /*                     (      C0 c0, ..., C(N-1) c(N-1)           ,       ) */                \
    DINLINE void operator()(T_Acc const& acc, Zone, BOOST_PP_ENUM_BINARY_PARAMS(N, C, c), const Functor& functor)     \
    {                                                                                                                 \
        const int dataVolume = math::CT::volume<typename Zone::Size>::type::value;                                    \
        const int blockVolume = math::CT::volume<BlockDim>::type::value;                                              \
                                                                                                                      \
        typedef typename math::Int<Zone::dim> PosType;                                                                \
        using namespace pmacc::algorithms::precisionCast;                                                             \
                                                                                                                      \
        for(int i = this->linearThreadIdx; i < dataVolume; i += blockVolume)                                          \
        {                                                                                                             \
            PosType pos = Zone::Offset::toRT()                                                                        \
                + precisionCast<typename PosType::type>(math::MapToPos<Zone::dim>()(typename Zone::Size(), i));       \
            functor(acc, BOOST_PP_ENUM(N, SHIFTACCESS_CURSOR, _));                                                    \
        }                                                                                                             \
    }

            /** Foreach algorithm that is executed by one cupla thread block
             *
             * \tparam BlockDim 3D compile-time vector (pmacc::math::CT::Int) of the size of the cupla blockDim.
             *
             * BlockDim could also be obtained from cupla itself at runtime but
             * it is faster to know it at compile-time.
             */
            template<typename BlockDim>
            struct Foreach
            {
            private:
                const int linearThreadIdx;

            public:
                DINLINE Foreach(int linearThreadIdx) : linearThreadIdx(linearThreadIdx)
                {
                }

                /* operator()(zone, cursor0, cursor1, ..., cursorN-1, functor or lambdaFun)
                 *
                 * \param zone compile-time zone object, see zone::CT::SphericZone. (e.g. ContainerType::Zone())
                 * \param cursorN cursor for the N-th data source (e.g. containerObj.origin())
                 * \param functor or lambdaFun either a functor with N arguments or a N-ary lambda function (e.g. _1 =
                 * _2)
                 *
                 * The functor or lambdaFun is called for each cell within the zone.
                 * It is called like functor(*cursor0(cellId), ..., *cursorN(cellId))
                 *
                 */
                BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(FOREACH_KERNEL_MAX_PARAMS), FOREACH_OPERATOR, _)
            };

#undef SHIFTACCESS_CURSOR
#undef FOREACH_OPERATOR

        } // namespace cuplaBlock
    } // namespace algorithm
} // namespace pmacc
