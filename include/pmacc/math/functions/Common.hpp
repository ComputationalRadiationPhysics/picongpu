/* Copyright 2024 Rene Widera
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

#include <alpaka/alpaka.hpp>

#include <type_traits>

namespace pmacc::math
{
    namespace detail
    {
        /** Trait to get alpaka math implementation type for given accelerator/math implementation type
         *
         * This trait is needed here as alpaka currently does not express this relation directly.
         * In alpaka it is only possible to get a type of implementation of a particular math function,
         * but not for all functions. So with existing alpaka traits e.g. Acc -> alpaka::math::AbsStdLib is
         * possible, but not Acc -> alpaka::math::MathStdLib.
         * And the latter is needed for math functions that call other math functions as described in #234.
         *
         * General implementation returns MathStdLib as it fits host-side usage and supported CPU
         * accelerators.
         *
         * @tparam T_AccOrMathImpl accelerator or math implementation type
         */
        template<typename T_AccOrMathImpl>
        struct MathImpl
        {
            using type = alpaka::math::MathStdLib;
        };

#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
        /** Trait to get alpaka math implementation type for CUDA accelerator
         *
         * @tparam T_Dim dimensionality of accelerator index space
         * @tparam T_Idx type of accelerator indexes
         */
        template<typename T_Dim, typename T_Idx>
        struct MathImpl<alpaka::AccGpuCudaRt<T_Dim, T_Idx>>
        {
            using type = alpaka::math::MathUniformCudaHipBuiltIn;
        };
#endif

#if ALPAKA_ACC_GPU_HIP_ENABLED == 1
        /** Trait to get alpaka math implementation type for HIP accelerator
         *
         * @tparam T_Dim dimensionality of accelerator index space
         * @tparam T_Idx type of accelerator indexes
         */
        template<typename T_Dim, typename T_Idx>
        struct MathImpl<alpaka::AccGpuHipRt<T_Dim, T_Idx>>
        {
            using type = alpaka::math::MathUniformCudaHipBuiltIn;
        };
#endif

        /** Get the concept implementation of the current accelerator
         *
         * @tparam T_AccOrMathImpl accelerator or math implementation [type alpaka::* or
         * alpaka::math::MathStdLib]
         * @tparam T_Concept alpaka concept
         * @return implementation of the concept
         */
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T_AccOrMathImpl, typename T_Concept>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto getConcept()
        {
            using ResultMathConcept = typename MathImpl<T_AccOrMathImpl>::type;

            using AccMathConcept = alpaka::concepts::ImplementationBase<T_Concept, Acc<DIM3>>;

            using AccThreadSeqMathConcept = alpaka::concepts::ImplementationBase<T_Concept, AccThreadSeq<DIM3>>;

            // cupla Acc and AccThreadSeq should use the same math concept implementation
            static_assert(
                std::is_same<AccMathConcept, AccThreadSeqMathConcept>::value,
                "The math concept implementation for the type 'Acc' and 'AccThreadSeq' must be equal");

            return ResultMathConcept{};
        }
    } // namespace detail

#define CUPLA_UNARY_MATH_FN_DETAIL(functionName, accOrMathImpl, alpakaMathConcept, alpakaMathTrait)                   \
    /**                                                                                                               \
     * @tparam T_Type argument type                                                                                   \
     * @param arg input argument                                                                                      \
     */                                                                                                               \
    ALPAKA_NO_HOST_ACC_WARNING                                                                                        \
    template<typename T_Type>                                                                                         \
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto functionName(                                                            \
        T_Type const& arg) /* return type is required for the compiler to detect host, device                         \
                            * function qualifier correctly                                                            \
                            */                                                                                        \
        ->decltype(alpaka::core::declval<alpaka::math::trait::alpakaMathTrait<                                        \
                       alpaka::concepts::ImplementationBase<alpakaMathConcept, accOrMathImpl>,                        \
                       T_Type>>()(detail::getConcept<accOrMathImpl, alpakaMathConcept>(), arg))                       \
    {                                                                                                                 \
        return alpaka::math::trait::alpakaMathTrait<                                                                  \
            alpaka::concepts::ImplementationBase<alpakaMathConcept, accOrMathImpl>,                                   \
            T_Type>{}(detail::getConcept<accOrMathImpl, alpakaMathConcept>(), arg);                                   \
    }

/* Using the free alpaka functions `alpaka::math::*` will result into `__host__ __device__`
 * errors, therefore the alpaka math trait must be used.
 */
#define CUPLA_BINARY_MATH_FN_DETAIL(functionName, accOrMathImpl, alpakaMathConcept, alpakaMathTrait)                  \
    /**                                                                                                               \
     * @tparam T_Type argument type                                                                                   \
     * @param arg1 first input argument                                                                               \
     * @param arg2 second input argument                                                                              \
     */                                                                                                               \
    ALPAKA_NO_HOST_ACC_WARNING                                                                                        \
    template<typename T_Type1, typename T_Type2>                                                                      \
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto functionName(                                                            \
        T_Type1 const& arg1,                                                                                          \
        T_Type2 const& arg2) /* return type is required for the compiler to detect host, device                       \
                              * function qualifier correctly                                                          \
                              */                                                                                      \
        ->decltype(alpaka::core::declval<alpaka::math::trait::alpakaMathTrait<                                        \
                       alpaka::concepts::ImplementationBase<alpakaMathConcept, accOrMathImpl>,                        \
                       T_Type1,                                                                                       \
                       T_Type2>>()(detail::getConcept<accOrMathImpl, alpakaMathConcept>(), arg1, arg2))               \
    {                                                                                                                 \
        return alpaka::math::trait::alpakaMathTrait<                                                                  \
            alpaka::concepts::ImplementationBase<alpakaMathConcept, accOrMathImpl>,                                   \
            T_Type1,                                                                                                  \
            T_Type2>{}(detail::getConcept<accOrMathImpl, alpakaMathConcept>(), arg1, arg2);                           \
    }

#if PMACC_DEVICE_COMPILE == 0
#    define CUPLA_UNARY_MATH_FN(functionName, alpakaMathConcept, alpakaMathTrait)                                     \
        CUPLA_UNARY_MATH_FN_DETAIL(functionName, alpaka::math::MathStdLib, alpakaMathConcept, alpakaMathTrait)
#    define CUPLA_BINARY_MATH_FN(functionName, alpakaMathConcept, alpakaMathTrait)                                    \
        CUPLA_BINARY_MATH_FN_DETAIL(functionName, alpaka::math::MathStdLib, alpakaMathConcept, alpakaMathTrait)
#else
#    define CUPLA_UNARY_MATH_FN(functionName, alpakaMathConcept, alpakaMathTrait)                                     \
        CUPLA_UNARY_MATH_FN_DETAIL(functionName, Acc<DIM1>, alpakaMathConcept, alpakaMathTrait)
#    define CUPLA_BINARY_MATH_FN(functionName, alpakaMathConcept, alpakaMathTrait)                                    \
        CUPLA_BINARY_MATH_FN_DETAIL(functionName, Acc<DIM1>, alpakaMathConcept, alpakaMathTrait)
#endif

} // namespace pmacc::math
