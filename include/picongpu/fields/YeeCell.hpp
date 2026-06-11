/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/Fields.def"
#include "picongpu/traits/FieldPosition.hpp"

#include <pmacc/math/Vector.hpp>

namespace picongpu
{
    namespace fields
    {
        /** classical Yee cell
         *
         * Defines staggered a cell where the magnetic and electric field are shifted by a half cell to each other.
         */
        struct YeeCell
        {
        };

    } // namespace fields

    namespace traits
    {
        /** position (float2_X) in cell for E_x, E_y, E_z
         */
        template<>
        struct FieldPosition<fields::YeeCell, FieldE, DIM2>
        {
            /** @tparam float2_X position of the component in the cell
             *  @tparam DIM3     Fields (E/B/J) have 3 components, even in 1 or 2D !
             */
            using VectorVector2D3V = ::pmacc::math::Vector<float2_X, DIM3> const;

            HDINLINE FieldPosition() = default;

            HDINLINE VectorVector2D3V operator()() const
            {
                float2_X const posE_x(0.5, 0.0);
                float2_X const posE_y(0.0, 0.5);
                float2_X const posE_z(0.0, 0.0);

                return VectorVector2D3V(posE_x, posE_y, posE_z);
            }
        };

        /** position (float3_X) in cell for E_x, E_y, E_z
         */
        template<>
        struct FieldPosition<fields::YeeCell, FieldE, DIM3>
        {
            /** @tparam float2_X position of the component in the cell
             *  @tparam DIM3     Fields (E/B/J) have 3 components, even in 1 or 2D !
             */
            using VectorVector3D3V = ::pmacc::math::Vector<float3_X, DIM3> const;

            HDINLINE FieldPosition() = default;

            HDINLINE VectorVector3D3V operator()() const
            {
                float3_X const posE_x(0.5, 0.0, 0.0);
                float3_X const posE_y(0.0, 0.5, 0.0);
                float3_X const posE_z(0.0, 0.0, 0.5);

                return VectorVector3D3V(posE_x, posE_y, posE_z);
            }
        };

        /** position (float2_X) in cell for B_x, B_y, B_z
         */
        template<>
        struct FieldPosition<fields::YeeCell, FieldB, DIM2>
        {
            /** @tparam float2_X position of the component in the cell
             *  @tparam DIM3     Fields (E/B/J) have 3 components, even in 1 or 2D !
             */
            using VectorVector2D3V = ::pmacc::math::Vector<float2_X, DIM3> const;

            HDINLINE FieldPosition() = default;

            HDINLINE VectorVector2D3V operator()() const
            {
                float2_X const posB_x(0.0, 0.5);
                float2_X const posB_y(0.5, 0.0);
                float2_X const posB_z(0.5, 0.5);

                return VectorVector2D3V(posB_x, posB_y, posB_z);
            }
        };

        /** position (float3_X) in cell for B_x, B_y, B_z
         */
        template<>
        struct FieldPosition<fields::YeeCell, FieldB, DIM3>
        {
            /** @tparam float2_X position of the component in the cell
             *  @tparam DIM3     Fields (E/B/J) have 3 components, even in 1 or 2D !
             */
            using VectorVector3D3V = ::pmacc::math::Vector<float3_X, DIM3> const;

            HDINLINE FieldPosition() = default;

            HDINLINE VectorVector3D3V operator()() const
            {
                float3_X const posB_x(0.0, 0.5, 0.5);
                float3_X const posB_y(0.5, 0.0, 0.5);
                float3_X const posB_z(0.5, 0.5, 0.0);

                return VectorVector3D3V(posB_x, posB_y, posB_z);
            }
        };

        /** position (floatD_X in case of T_simDim == simDim) in cell for
         *  J_x, J_y, J_z
         */
        template<uint32_t T_simDim>
        struct FieldPosition<fields::YeeCell, FieldJ, T_simDim>
            : public FieldPosition<fields::YeeCell, FieldE, T_simDim>
        {
            HDINLINE FieldPosition() = default;
        };

        /** position (floatD_X in case of T_simDim == simDim) in cell, wrapped in
         * one-component vector since it's a scalar field with only one component, for the
         * scalar field FieldTmp
         */
        template<uint32_t T_simDim>
        struct FieldPosition<fields::YeeCell, FieldTmp, T_simDim>
        {
            using FieldPos = pmacc::math::Vector<float_X, T_simDim>;
            using ReturnType = pmacc::math::Vector<FieldPos, DIM1>;

            HDINLINE FieldPosition() = default;

            HDINLINE ReturnType operator()() const
            {
                return ReturnType(FieldPos::create(0.0));
            }
        };

    } // namespace traits
} // namespace picongpu
