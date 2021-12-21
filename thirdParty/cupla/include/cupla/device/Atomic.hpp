/* Copyright 2020 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */


#pragma once

#include "cupla/datatypes/uint.hpp"
#include "cupla/device/Hierarchy.hpp"
#include "cupla/types.hpp"

#include <alpaka/alpaka.hpp>

namespace cupla
{
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{
inline namespace device
{

#define CUPLA_UNARY_ATOMIC_OP(functionName, alpakaOp)                          \
        /*!                                                                    \
         * Compared to their CUDA/HIP counterparts, these functions take an additional last \
         * parameter to denote atomicity (synchronization) level. This parameter is \
         * of type cupla::hierarchy::{Grids|Blocks|Threads}. Grids corresponds \
         * to atomicity between different kernels, Blocks - to different blocks \
         * in the same grid/kernel, Threads - to threads of the same block.    \
         * @tparam T_Hierarchy parallelism hierarchy level within the operation is atomic [type cupla::hierarchy::*] \
         * @tparam T_Acc alpaka accelerator [alpaka::*]                   \
         * @tparam T_Type type of the value                                    \
         * @param acc alpaka accelerator                                       \
         * @param ptr destination pointer                                      \
         * @param value source value                                           \
         * @{                                                                  \
         */                                                                    \
        template<                                                              \
            typename T_Hierarchy,                                              \
            typename T_Acc,                                                    \
            typename T_Type                                                    \
        >                                                                      \
        ALPAKA_FN_ACC ALPAKA_FN_INLINE                                         \
        T_Type functionName(                                                   \
            T_Acc const & acc,                                                 \
            T_Type *ptr,                                                       \
            T_Type const & value                                               \
        )                                                                      \
        {                                                                      \
            return ::alpaka::atomicOp< alpakaOp >(                     \
                acc,                                                           \
                ptr,                                                           \
                value,                                                         \
                T_Hierarchy{}                                                  \
            );                                                                 \
        }                                                                      \
                                                                               \
        /*! @param hierarchy hierarchy level within the operation is atomic    \
         */                                                                    \
        template<                                                              \
            typename T_Acc,                                                    \
            typename T_Type,                                                   \
            typename T_Hierarchy = alpaka::hierarchy::Grids                    \
        >                                                                      \
        ALPAKA_FN_ACC ALPAKA_FN_INLINE                                         \
        T_Type functionName(                                                   \
            T_Acc const & acc,                                                 \
            T_Type *ptr,                                                       \
            T_Type const & value,                                              \
            T_Hierarchy const & hierarchy = T_Hierarchy()                      \
        )                                                                      \
        {                                                                      \
            return functionName< T_Hierarchy >(                                \
                acc,                                                           \
                ptr,                                                           \
                value                                                          \
            );                                                                 \
        }                                                                      \
        /*!@}                                                                  \
         */

        /// atomic addition
        CUPLA_UNARY_ATOMIC_OP( atomicAdd, ::alpaka::AtomicAdd )
        /// atomic subtraction
        CUPLA_UNARY_ATOMIC_OP( atomicSub, ::alpaka::AtomicSub )
        /// atomic minimum
        CUPLA_UNARY_ATOMIC_OP( atomicMin, ::alpaka::AtomicMin )
        /// atomic maximum
        CUPLA_UNARY_ATOMIC_OP( atomicMax, ::alpaka::AtomicMax )
        /// atomic increment
        CUPLA_UNARY_ATOMIC_OP( atomicInc, ::alpaka::AtomicInc )
        /// atomic decrement
        CUPLA_UNARY_ATOMIC_OP( atomicDec, ::alpaka::AtomicDec )
        /// atomic bit-wise and
        CUPLA_UNARY_ATOMIC_OP( atomicAnd, ::alpaka::AtomicAnd )
        /// atomic bit-wise or
        CUPLA_UNARY_ATOMIC_OP( atomicOr, ::alpaka::AtomicOr )
        /// atomic exchange
        CUPLA_UNARY_ATOMIC_OP( atomicExch, ::alpaka::AtomicExch )
        /// atomic bit-wise xor
        CUPLA_UNARY_ATOMIC_OP( atomicXor, ::alpaka::AtomicXor )

#undef CUPLA_UNARY_ATOMIC_OP

        /** atomic compare and swap
         *
         * @{
         * @tparam T_Hierarchy parallelism hierarchy level within the operation is atomic [type cupla::hierarchy::*]
         * @tparam T_Acc alpaka accelerator [alpaka::*]
         * @tparam T_Type type of the value
         * @param acc alpaka accelerator
         * @param ptr destination pointer
         * @param value source value
         */
        template<
            typename T_Hierarchy,
            typename T_Acc,
            typename T_Type
        >
        ALPAKA_FN_ACC ALPAKA_FN_INLINE
        T_Type atomicCas(
            T_Acc const & acc,
            T_Type *ptr,
            T_Type const & compare,
            T_Type const & value
        )
        {
            return ::alpaka::atomicOp< ::alpaka::AtomicCas >(
                acc,
                ptr,
                compare,
                value,
                T_Hierarchy{}
            );
        }

        /*! @param hierarchy hierarchy level within the operation is atomic
         */
        template<
            typename T_Acc,
            typename T_Type,
            typename T_Hierarchy = hierarchy::Grids
        >
        ALPAKA_FN_ACC ALPAKA_FN_INLINE
        T_Type atomicCas(
            T_Acc const & acc,
            T_Type *ptr,
            T_Type const & compare,
            T_Type const & value,
            T_Hierarchy const & hierarchy = T_Hierarchy()
        )
        {
            return atomicCas< T_Hierarchy >(
                acc,
                ptr,
                compare,
                value
            );
        }
        /*!@}
         */

} // namespace device
} // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namespace cupla
