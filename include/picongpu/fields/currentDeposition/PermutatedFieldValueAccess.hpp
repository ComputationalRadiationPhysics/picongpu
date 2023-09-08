/* Copyright 2022-2023 Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pmacc/math/Vector.hpp>
#include <pmacc/types.hpp>


namespace picongpu
{
    namespace currentSolver
    {
        /** Permute the field dimensions and the vector value access
         *
         * @tparam T_DataBox dataBox type, ValueType must be a pmacc::math::Vector<>
         * @tparam T_PermutationVector compile-time vector (pmacc::math::CT::Int) that describes the mapping.
         *                             x-axis -> T_Permutation::at<0>, y-axis -> T_Permutation::at<1>, ...
         */
        template<typename T_DataBox, typename T_PermutationVector>
        class PermutatedFieldValueAccess
        {
            T_DataBox m_dataBox;

        public:
            /** constructor
             *
             * @param dataBox input data box to permute
             */
            HDINLINE PermutatedFieldValueAccess(T_DataBox& dataBox) : m_dataBox(dataBox)
            {
            }

            /** Get the permuted component of the vector field
             *
             *
             * The idx to select a value within the vector field and the component of the vector type is
             * automatically permuted.
             *
             * @param idx index to access within the vector field
             *
             * @{
             * @tparam T_component component of the ValueType
             * @return component of the dataBox ValueType
             */
            template<uint32_t T_component>
            HDINLINE decltype(auto) get(DataSpace<DIM2> const& idx)
            {
                constexpr uint32_t dim = T_DataBox::Dim;
                static_assert(dim == DIM2);

                constexpr auto x = T_PermutationVector::x::value;
                constexpr auto y = T_PermutationVector::y::value;

                using ValueComponentIdx = typename T_PermutationVector::template at<T_component>::type;
                constexpr auto valueComponentIdx = ValueComponentIdx::value;

                DataSpace<DIM2> permutatedIdx;
                permutatedIdx[x] = idx.x();
                permutatedIdx[y] = idx.y();

                return m_dataBox(permutatedIdx)[valueComponentIdx];
            }

            template<uint32_t T_component>
            HDINLINE decltype(auto) get(DataSpace<DIM3> const& idx)
            {
                constexpr uint32_t dim = T_DataBox::Dim;
                static_assert(dim == DIM3);

                constexpr auto x = T_PermutationVector::x::value;
                constexpr auto y = T_PermutationVector::y::value;
                constexpr auto z = T_PermutationVector::z::value;

                using ValueComponentIdx = typename T_PermutationVector::template at<T_component>::type;
                constexpr auto valueComponentIdx = ValueComponentIdx::value;

                /** @todo rewrite this as gather method instead of a scatter
                 * - for x component, search for the index (i0) where 0 is in T_PermutationVector
                 * - assign the value of permutatedIdx[0] = idx[i0], permutatedIdx.[1] = idx[i1], ...
                 */
                DataSpace<DIM3> permutatedIdx;
                permutatedIdx[x] = idx.x();
                permutatedIdx[y] = idx.y();
                permutatedIdx[z] = idx.z();
                return m_dataBox(permutatedIdx)[valueComponentIdx];
            }

            template<uint32_t T_component>
            HDINLINE decltype(auto) get(int x, int y)
            {
                return get<T_component>(DataSpace<DIM2>(x, y));
            }

            template<uint32_t T_component>
            HDINLINE decltype(auto) get(int x, int y, int z)
            {
                return get<T_component>(DataSpace<DIM3>(x, y, z));
            }
            /** @} */
        };


        /** creates an permuted field accessor
         *
         * The field dimension access and the value acces is permuted. @see PermutatedFieldValueAccess
         *
         * @tparam T_DataBox dataBox type, ValueType must be a pmacc::math::Vector<>
         * @tparam T_PermutationVector compile-time vector (pmacc::math::CT::Int) that describes the mapping.
         *                             x-axis -> T_Permutation::at<0>, y-axis -> T_Permutation::at<1>, ...
         * @param dataBox input data box to permute
         * @return permuted access to the input field data
         */
        template<typename T_PermutationVector, typename T_DataBox>
        HDINLINE auto makePermutatedFieldValueAccess(T_DataBox& dataBox)
        {
            return PermutatedFieldValueAccess<T_DataBox, T_PermutationVector>{dataBox};
        }
    } // namespace currentSolver
} // namespace picongpu
