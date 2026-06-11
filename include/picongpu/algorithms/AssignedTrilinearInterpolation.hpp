/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/attribute/unroll.hpp>
#include <pmacc/memory/Array.hpp>
#include <pmacc/traits/GetStringProperties.hpp>
#include <pmacc/types.hpp>

#include <type_traits>

namespace picongpu
{
    struct AssignedTrilinearInterpolation
    {
        /** Does a trilinear field-to-point interpolation for
         * arbitrary assignment function and arbitrary field_value types.
         *
         * @tparam T_begin lower margin for interpolation
         * @tparam T_end upper margin for interpolation
         * @tparam T_FieldAccessorFunctor type of the field access functor supporting a initializer list of simdDim
         *                                indices
         * @tparam T_AssignmentFunction type of shape functors
         *
         * @param fieldAccess field access method pointing to the particle located cell
         * @param shapeFunctors Array with d shape functors, where d is the dimensionality of the field represented by
         *                      cursor. The shape functor must have the interface to call
         *                      `operator()(relative_grid_point)` and return the assignment value for the given grid
         *                      point.
         * @return sum over: field_value * assignment
         *
         * interpolate on grid points in range [T_begin;T_end]
         *
         * @{
         */
        template<int T_begin, int T_end, typename T_FieldAccessorFunctor, typename T_AssignmentFunction>
        HDINLINE static auto interpolate(
            T_FieldAccessorFunctor const& fieldAccess,
            pmacc::memory::Array<T_AssignmentFunction, 3> const& shapeFunctors)
        {
            [[maybe_unused]] constexpr auto iterations = T_end - T_begin + 1;

            using type = decltype(fieldAccess({0, 0, 0}) * shapeFunctors[0](0));

            /* The implementation assumes that x is the fastest moving index to iterate over contiguous memory
             * e.g. a row, to optimize memory fetch operations.
             */
            auto result_z = type(0.0);
            PMACC_UNROLL(iterations)
            for(int z = T_begin; z <= T_end; ++z)
            {
                auto result_y = type(0.0);
                PMACC_UNROLL(iterations)
                for(int y = T_begin; y <= T_end; ++y)
                {
                    auto result_x = type(0.0);
                    PMACC_UNROLL(iterations)
                    for(int x = T_begin; x <= T_end; ++x)
                        /* a form factor is the "amount of particle" that is affected by this cell
                         * so we have to sum over: cell_value * form_factor
                         */
                        result_x += fieldAccess({x, y, z}) * shapeFunctors[0](x);

                    result_y += result_x * shapeFunctors[1](y);
                }
                result_z += result_y * shapeFunctors[2](z);
            }
            return result_z;
        }

        /** Implementation for 2D position*/
        template<int T_begin, int T_end, class T_FieldAccessorFunctor, class T_AssignmentFunction>
        HDINLINE static auto interpolate(
            T_FieldAccessorFunctor const& fieldAccess,
            pmacc::memory::Array<T_AssignmentFunction, 2> const& shapeFunctors)
        {
            [[maybe_unused]] constexpr int iterations = T_end - T_begin + 1;

            using type = decltype(fieldAccess({0, 0}) * shapeFunctors[0](0));
            /* The implementation assumes that x is the fastest moving index to iterate over contiguous memory
             * e.g. a row, to optimize memory fetch operations.
             */
            auto result_y = type(0.0);
            PMACC_UNROLL(iterations)
            for(int y = T_begin; y <= T_end; ++y)
            {
                auto result_x = type(0.0);
                PMACC_UNROLL(iterations)
                for(int x = T_begin; x <= T_end; ++x)
                    // a form factor is the "amount of particle" that is affected by this cell
                    // so we have to sum over: cell_value * form_factor
                    result_x += fieldAccess({x, y}) * shapeFunctors[0](x);

                result_y += result_x * shapeFunctors[1](y);
            }
            return result_y;
        }

        static auto getStringProperties() -> pmacc::traits::StringProperty
        {
            pmacc::traits::StringProperty propList("name", "uniform");
            return propList;
        }
    };

    //! @}

} // namespace picongpu
