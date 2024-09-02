/* Copyright 2013-2023 Heiko Burau, Rene Widera, Benjamin Worpitz
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/math/vector/Vector.hpp"
#include "pmacc/types.hpp"

#include <boost/mpl/aux_/na.hpp>

#include <cstdint>

namespace pmacc
{
    namespace math
    {
        namespace CT
        {
            namespace detail
            {
                struct na
                {
                };

                template<typename Arg0>
                struct TypeSelector
                {
                    using type = Arg0;
                };

                /** get integral type*/
                template<typename T, T value>
                struct TypeSelector<std::integral_constant<T, value>>
                {
                    using type = T;
                };

                template<>
                struct TypeSelector<detail::na>
                {
                    using type = mp_int<0u>;
                };

            } // namespace detail

            template<typename Arg0 = detail::na, typename Arg1 = detail::na, typename Arg2 = detail::na>
            struct Vector
            {
                using x = Arg0;
                using y = Arg1;
                using z = Arg2;

                using mplVector = mp_remove<mp_list<x, y, z>, detail::na>;

                template<uint32_t element>
                struct at
                {
                    using type = mp_at_c<mplVector, element>;
                };

                static constexpr uint32_t dim = mp_size<mplVector>::value;

                using type = typename detail::TypeSelector<x>::type;
                using This = Vector<x, y, z>;
                using RT_type = math::Vector<type, dim>;
                using vector_type = This;

                template<typename OtherType>
                HDINLINE operator math::Vector<OtherType, dim>() const
                {
                    return toRT();
                }

                /** Create a runtime Vector
                 *
                 *  Creates the corresponding runtime vector object.
                 *
                 *  @return RT_type runtime vector with same value type
                 */
                template<uint32_t T_deferDim = dim, std::enable_if_t<T_deferDim == 1u, int> = 0>
                static constexpr RT_type toRT()
                {
                    return RT_type(This::x::value);
                }
                template<uint32_t T_deferDim = dim, std::enable_if_t<T_deferDim == 2u, int> = 0>
                static constexpr RT_type toRT()
                {
                    return RT_type(This::x::value, This::y::value);
                }
                template<uint32_t T_deferDim = dim, std::enable_if_t<T_deferDim == 3u, int> = 0>
                static constexpr RT_type toRT()
                {
                    return RT_type(This::x::value, This::y::value, This::z::value);
                }
            };

            //*********************************************************

            //________________________OperatorBase____________________________

            template<typename Lhs, typename Rhs, template<typename...> typename T_BinaryOperator>
            struct applyOperator
            {
                using type =
                    typename applyOperator<typename Lhs::vector_type, typename Rhs::vector_type, T_BinaryOperator>::
                        type;
            };

            template<typename T_TypeA, typename T_TypeB, template<typename...> typename T_BinaryOperator>
            struct applyOperator<CT::Vector<T_TypeA>, CT::Vector<T_TypeB>, T_BinaryOperator>
            {
                using OpResult = T_BinaryOperator<T_TypeA, T_TypeB>;
                using type = CT::Vector<OpResult>;
            };

            template<
                typename T_TypeA0,
                typename T_TypeA1,
                typename T_TypeB0,
                typename T_TypeB1,
                template<typename...>
                typename T_BinaryOperator>
            struct applyOperator<CT::Vector<T_TypeA0, T_TypeA1>, CT::Vector<T_TypeB0, T_TypeB1>, T_BinaryOperator>
            {
                using OpResult0 = T_BinaryOperator<T_TypeA0, T_TypeB0>;
                using OpResult1 = T_BinaryOperator<T_TypeA1, T_TypeB1>;
                using type = CT::Vector<OpResult0, OpResult1>;
            };

            template<
                typename T_TypeA0,
                typename T_TypeA1,
                typename T_TypeA2,
                typename T_TypeB0,
                typename T_TypeB1,
                typename T_TypeB2,
                template<typename...>
                typename T_BinaryOperator>
            struct applyOperator<
                CT::Vector<T_TypeA0, T_TypeA1, T_TypeA2>,
                CT::Vector<T_TypeB0, T_TypeB1, T_TypeB2>,
                T_BinaryOperator>
            {
                using OpResult0 = T_BinaryOperator<T_TypeA0, T_TypeB0>;
                using OpResult1 = T_BinaryOperator<T_TypeA1, T_TypeB1>;
                using OpResult2 = T_BinaryOperator<T_TypeA2, T_TypeB2>;
                using type = CT::Vector<OpResult0, OpResult1, OpResult2>;
            };

            template<typename A, typename B>
            using mp_times = std::integral_constant<decltype(A::value * B::value), A::value * B::value>;

            //________________________A D D____________________________

            template<typename Lhs, typename Rhs>
            struct add
            {
                using type =
                    typename applyOperator<typename Lhs::vector_type, typename Rhs::vector_type, mp_plus>::type;
            };

            //________________________M U L____________________________

            template<typename Lhs, typename Rhs>
            struct mul
            {
                using type =
                    typename applyOperator<typename Lhs::vector_type, typename Rhs::vector_type, mp_times>::type;
            };

            //________________________M A X____________________________

            /** maximum value
             *
             * @tparam Lhs input vector
             * @tparam Rhs input vector
             * @return ::type if Rhs is not given - maximum value in elements of Lhs else
             *         vector with point-wise maximum value per component
             */
            template<typename Lhs, typename Rhs = void>
            struct max
            {
                using type =
                    typename applyOperator<typename Lhs::vector_type, typename Rhs::vector_type, mp_max>::type;
            };


            /** get element with maximum value
             *
             * @tparam T_Vec input vector
             * @return ::type maximum value in elements of T_Vec
             */
            template<typename T_Vec>
            struct max<T_Vec, void>
            {
                using type = mp_apply<mp_max, typename T_Vec::mplVector>;
            };

            //________________________M I N____________________________


            /** minimum value
             *
             * @tparam Lhs input vector
             * @tparam Rhs input vector
             * @return ::type if Rhs is not given - minimum value in elements of Lhs else
             *         vector with point-wise minimum value per component
             */
            template<typename Lhs, typename Rhs = void>
            struct min
            {
                using type =
                    typename applyOperator<typename Lhs::vector_type, typename Rhs::vector_type, mp_min>::type;
            };

            /** get element with minimum value
             *
             * @tparam T_Vec input vector
             * @return ::type minimum value in elements of T_Vec
             */
            template<typename T_Vec>
            struct min<T_Vec, void>
            {
                using type = mp_apply<mp_min, typename T_Vec::mplVector>;
            };

            //________________________D O T____________________________

            template<typename Lhs, typename Rhs>
            struct dot
            {
                using MulResult = typename mul<Lhs, Rhs>::type;
                using type = mp_fold<typename MulResult::mplVector, mp_int<0>, mp_plus>;
            };

            //________________________V O L U M E____________________________

            template<typename T_Vec>
            struct volume
            {
                using type = mp_fold<typename T_Vec::mplVector, mp_int<1>, mp_times>;
            };

            //________________________S H R I N K T O________________________

            /** shrink CT vector to given component count (dimension)
             *
             * This operation is designed to handle vectors with up to 3 components
             *
             * @tparam T_Vec vector to shrink
             * @tparam T_dim target component count
             * @treturn ::type new shrinked vector
             */
            template<typename T_Vec, uint32_t T_dim>
            struct shrinkTo;

            template<typename T_Vec>
            struct shrinkTo<T_Vec, DIM3>
            {
                using Vec = T_Vec;
                using type = CT::Vector<typename Vec::x, typename Vec::y, typename Vec::z>;
            };

            template<typename T_Vec>
            struct shrinkTo<T_Vec, DIM2>
            {
                using Vec = T_Vec;
                using type = CT::Vector<typename Vec::x, typename Vec::y>;
            };

            template<typename T_Vec>
            struct shrinkTo<T_Vec, DIM1>
            {
                using Vec = T_Vec;
                using type = CT::Vector<typename Vec::x>;
            };

            //________________________A S S I G N________________________

            /** Assign a type to a given component in the CT::Vector
             *
             * defines a public type as result
             *
             * @tparam T_Vec math::CT::Vector which should be changed
             * @tparam T_ComponentPos number of component to changed (type must be std::integral_constant<anyType,X>)
             * @tparam T_Value new value
             */
            template<typename T_Vec, typename T_ComponentPos, typename T_Value>
            struct Assign;

            template<typename T_Value, typename T_0, typename T_1, typename T_2, typename T_IntegralType>
            struct Assign<pmacc::math::CT::Vector<T_0, T_1, T_2>, std::integral_constant<T_IntegralType, 0>, T_Value>
            {
                using type = pmacc::math::CT::Vector<T_Value, T_1, T_2>;
            };

            template<typename T_Value, typename T_0, typename T_1, typename T_2, typename T_IntegralType>
            struct Assign<pmacc::math::CT::Vector<T_0, T_1, T_2>, std::integral_constant<T_IntegralType, 1>, T_Value>
            {
                using type = pmacc::math::CT::Vector<T_0, T_Value, T_2>;
            };

            template<typename T_Value, typename T_0, typename T_1, typename T_2, typename T_IntegralType>
            struct Assign<pmacc::math::CT::Vector<T_0, T_1, T_2>, std::integral_constant<T_IntegralType, 2>, T_Value>
            {
                using type = pmacc::math::CT::Vector<T_0, T_1, T_Value>;
            };

            /** Assign a type to a given component in the CT::Vector if position is not out of range
             *
             * if T_ComponentPos < T_Vec::dim ? T_Value is assigned to component T_ComponentPos
             * else nothing is done.
             * defines a public type as result
             *
             * @tparam T_Vec math::CT::Vector which should be changed
             * @tparam T_ComponentPos number of component to changed (type must be std::integral_constant<anyType,X>)
             * @tparam T_Value new value
             */
            template<typename T_Vec, typename T_ComponentPos, typename T_Value>
            struct AssignIfInRange
            {
                using type = mp_if_c
                    < T_ComponentPos::value<T_Vec::dim, typename Assign<T_Vec, T_ComponentPos, T_Value>::type, T_Vec>;
            };

            //________________________At_c____________________________

            /** get element from a CT::Vector
             *
             * defines a public type as result
             *
             * @tparam T_Vec input CT::Vector
             * @tparam T_idx integral index of the component
             */
            template<typename T_Vec, size_t T_idx>
            struct At_c
            {
                using type = mp_at_c<typename T_Vec::mplVector, T_idx>;
            };

            //________________________At____________________________

            /** get element from a CT::Vector
             *
             * defines a public type as result
             *
             * @tparam T_Vec input CT::Vector
             * @tparam T_Idx integral type index of the component (e.g. pmacc::mp_int<2>)
             */
            template<typename T_Vec, typename T_Idx>
            struct At
            {
                using type = mp_at<typename T_Vec::mplVector, T_Idx>;
            };

            //________________________make_Vector___________________

            /** create CT::Vector with equal elements
             *
             * defines a public type as result
             *
             * @tparam T_dim count of components
             * @tparam T_Type type which is assigned to all components
             */
            template<uint32_t T_dim, typename T_Type>
            struct make_Vector;

            template<typename T_Type>
            struct make_Vector<1, T_Type>
            {
                using type = pmacc::math::CT::Vector<T_Type>;
            };

            template<typename T_Type>
            struct make_Vector<2, T_Type>
            {
                using type = pmacc::math::CT::Vector<T_Type, T_Type>;
            };

            template<typename T_Type>
            struct make_Vector<3, T_Type>
            {
                using type = pmacc::math::CT::Vector<T_Type, T_Type, T_Type>;
            };

            //________________________make_BasisVector___________________

            /** Create CT::Vector that is the unit basis vector along the given direction
             *
             * Defines a public type as result.
             * In case 0 <= T_direction < T_dim, return the basis vector type with value
             * 1 in component T_direction and 0 in other components, otherwise return the
             * zero vector type.
             *
             * @tparam T_dim count of components
             * @tparam T_direction index of the basis vector direction
             * @tparam T_ValueType value type of the vector
             */
            template<uint32_t T_dim, uint32_t T_direction, typename T_ValueType = int>
            struct make_BasisVector
            {
                using Zeroes = typename make_Vector<T_dim, std::integral_constant<T_ValueType, 0u>>::type;
                using type = typename AssignIfInRange<
                    Zeroes,
                    std::integral_constant<size_t, T_direction>,
                    std::integral_constant<T_ValueType, 1>>::type;
            };

        } // namespace CT
    } // namespace math
} // namespace pmacc
