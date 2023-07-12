/* Copyright 2023 Brian Marre, Rene Widera
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

#include "pmacc/attribute/unroll.hpp"
#include "pmacc/math/Vector.hpp"
#include "pmacc/memory/Array.hpp"

#include <cstdint>

namespace pmacc::math
{
    /** common stuff general interface
     *
     * @tparam T_Type data type used for storage of elements
     * @tparam T_order number of dimensions of matrix(so called order),
     *      for example MxN matrix has order 2
     * @tparam T_extent pmacc::math::CT::Vector, extent of matrix in each dimensions
     */
    template<typename T_Type, typename T_Extent>
    struct Matrix
    {
        using Extent = T_Extent;
        using Type = T_Type;

        static auto constexpr m = Extent::template at<0u>::type::value;
        static auto constexpr n = Extent::template at<1u>::type::value;

        using S_Elements = pmacc::memory::Array<T_Type, CT::volume<T_Extent>::type::value>;
        using Idx = typename S_Elements::size_type;

    private:
        S_Elements elements;

        /** get linear memory storage index from n-dimensional index
         *
         * indexation with the following scheme for a matrix with extent = < <N0>, <N1> >
         *
         * #(linear Index) | 0 | 1 | ... | <N0>-1 | <N0> | <N0> + 1 | ... | <N0>+<N0>-1 | 2 * <N0>| ...
         * ----------------|---|---|-----|--------|------|----------|-----|-------------|---------|----
         *                m| 0 | 1 | ... | <N0>-1 |    0 |        1 | ... |  (<N0> - 1) | 0, ...  | ...
         *                n| 0 | 0 | ... |      0 |    1 |        1 | ... |           1 | 2, ...  | ...
         *
         * @attention no range checks outside debug, invalid input will lead to illegal memory access!
         */
        HDINLINE static Idx getLinearIndex(uint32_t const& mIdx, uint32_t const& nIdx)
        {
#ifndef NDEBUG
            // debug range check
            if(mIdx >= m)
            {
                printf("PMACC_ERROR: invalid mIdx for matrix access!\n");
                return static_cast<uint32_t>(0u);
            }
            if(nIdx >= n)
            {
                printf("PMACC_ERROR: invalid nIdx for matrix access!\n");
                return static_cast<uint32_t>(0u);
            }
#endif
            return mIdx + nIdx * m;
        }

    public:
        /** constructor
         *
         * initialize each member with the given value
         *
         * @param value value assigned to each matrix entry
         */
        HDINLINE explicit Matrix(T_Type const& value) : elements(value)
        {
            PMACC_CASSERT_MSG(not_a_matrix, T_Extent::dim == 2u);
        }

        HDINLINE Matrix(pmacc::math::Vector<T_Type, m> const& vector)
        {
            PMACC_CASSERT(n == 1u);

            for(uint32_t i = 0u; i < m; ++i)
                (*this)(i, 0u) = vector[i];
        }

        /** access via (m, n)
         *
         * @attention idx indexation starts with 0!
         * @attention no range checks outside debug compile!, invalid idx will result in illegal memory access!
         *  @{
         */
        HDINLINE T_Type& operator()(uint32_t const mIdx, uint32_t const nIdx)
        {
            return elements[getLinearIndex(mIdx, nIdx)];
        }

        HDINLINE T_Type const& operator()(uint32_t const mIdx, uint32_t const nIdx) const
        {
            return elements[getLinearIndex(mIdx, nIdx)];
        }

        /** @} */

        //! matrix multiplication, C = A x B =^= [C = A.mMul(B)]
        template<typename T_ExtentRhs>
        HDINLINE Matrix<T_Type, pmacc::math::CT::UInt32<m, Matrix<T_Type, T_ExtentRhs>::n>> mMul(
            Matrix<T_Type, T_ExtentRhs> const& rhs) const
        {
            using MatrixB = Matrix<T_Type, T_ExtentRhs>;
            using ExtentC = pmacc::math::CT::UInt32<m, MatrixB::n>;

            Matrix<T_Type, ExtentC> result(static_cast<T_Type>(0));

            PMACC_CASSERT(n == MatrixB::m);

            constexpr auto jExtent = MatrixB::n;
            PMACC_UNROLL(jExtent)
            for(uint32_t j = 0u; j < jExtent; ++j)
            {
                constexpr auto iExtent = m;
                PMACC_UNROLL(iExtent)
                for(uint32_t i = 0u; i < iExtent; ++i)
                {
                    T_Type& c_ij = result(i, j);

                    constexpr auto nExtent = n;
                    PMACC_UNROLL(nExtent)
                    for(uint32_t k = 0u; k < nExtent; ++k)
                    {
                        c_ij += (*this)(i, k) * rhs(k, j);
                    }
                }
            }

            return result;
        }

        //! component wise (scalar) multiplication
        HDINLINE Matrix operator*(T_Type const a) const
        {
            Matrix result(*this);
            for(uint32_t j = 0u; j < n; ++j)
            {
                for(uint32_t i = 0u; i < m; ++i)
                {
                    result(i, j) *= a;
                }
            }
            return result;
        }

        HDINLINE Matrix operator+(Matrix const& rhs) const
        {
            Matrix result(*this);

            for(uint32_t j = 0u; j < n; ++j)
            {
                for(uint32_t i = 0u; i < m; ++i)
                {
                    result(i, j) += rhs(i, j);
                }
            }
            return result;
        }

        HDINLINE Matrix& operator=(Matrix const& rhs) = default;
    };
} // namespace pmacc::math
