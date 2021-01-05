/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Sergei Bastrakov
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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/traits/FieldPosition.hpp"
#include "picongpu/fields/Fields.def"

#include <pmacc/math/Vector.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace cellType
        {
            struct Yee
            {
            };

        } // namespace cellType
    } // namespace fields

    namespace traits
    {
        /** position (float2_X) in cell for E_x, E_y, E_z
         */
        template<>
        struct FieldPosition<fields::cellType::Yee, FieldE, DIM2>
        {
            /** \tparam float2_X position of the component in the cell
             *  \tparam DIM3     Fields (E/B/J) have 3 components, even in 1 or 2D !
             */
            using VectorVector2D3V = const ::pmacc::math::Vector<float2_X, DIM3>;
            /// boost::result_of hints
            template<class>
            struct result;

            template<class F>
            struct result<F()>
            {
                using type = VectorVector2D3V;
            };

            HDINLINE FieldPosition()
            {
            }

            HDINLINE VectorVector2D3V operator()() const
            {
                const float2_X posE_x(0.5, 0.0);
                const float2_X posE_y(0.0, 0.5);
                const float2_X posE_z(0.0, 0.0);

                return VectorVector2D3V(posE_x, posE_y, posE_z);
            }
        };

        /** position (float3_X) in cell for E_x, E_y, E_z
         */
        template<>
        struct FieldPosition<fields::cellType::Yee, FieldE, DIM3>
        {
            /** \tparam float2_X position of the component in the cell
             *  \tparam DIM3     Fields (E/B/J) have 3 components, even in 1 or 2D !
             */
            using VectorVector3D3V = const ::pmacc::math::Vector<float3_X, DIM3>;

            /// boost::result_of hints
            template<class>
            struct result;

            template<class F>
            struct result<F()>
            {
                using type = VectorVector3D3V;
            };

            HDINLINE FieldPosition()
            {
            }

            HDINLINE VectorVector3D3V operator()() const
            {
                const float3_X posE_x(0.5, 0.0, 0.0);
                const float3_X posE_y(0.0, 0.5, 0.0);
                const float3_X posE_z(0.0, 0.0, 0.5);

                return VectorVector3D3V(posE_x, posE_y, posE_z);
            }
        };

        /** position (float2_X) in cell for B_x, B_y, B_z
         */
        template<>
        struct FieldPosition<fields::cellType::Yee, FieldB, DIM2>
        {
            /** \tparam float2_X position of the component in the cell
             *  \tparam DIM3     Fields (E/B/J) have 3 components, even in 1 or 2D !
             */
            using VectorVector2D3V = const ::pmacc::math::Vector<float2_X, DIM3>;
            /// boost::result_of hints
            template<class>
            struct result;

            template<class F>
            struct result<F()>
            {
                using type = VectorVector2D3V;
            };

            HDINLINE FieldPosition()
            {
            }

            HDINLINE VectorVector2D3V operator()() const
            {
                const float2_X posB_x(0.0, 0.5);
                const float2_X posB_y(0.5, 0.0);
                const float2_X posB_z(0.5, 0.5);

                return VectorVector2D3V(posB_x, posB_y, posB_z);
            }
        };

        /** position (float3_X) in cell for B_x, B_y, B_z
         */
        template<>
        struct FieldPosition<fields::cellType::Yee, FieldB, DIM3>
        {
            /** \tparam float2_X position of the component in the cell
             *  \tparam DIM3     Fields (E/B/J) have 3 components, even in 1 or 2D !
             */
            using VectorVector3D3V = const ::pmacc::math::Vector<float3_X, DIM3>;

            /// boost::result_of hints
            template<class>
            struct result;

            template<class F>
            struct result<F()>
            {
                using type = VectorVector3D3V;
            };

            HDINLINE FieldPosition()
            {
            }

            HDINLINE VectorVector3D3V operator()() const
            {
                const float3_X posB_x(0.0, 0.5, 0.5);
                const float3_X posB_y(0.5, 0.0, 0.5);
                const float3_X posB_z(0.5, 0.5, 0.0);

                return VectorVector3D3V(posB_x, posB_y, posB_z);
            }
        };

        /** position (floatD_X in case of T_simDim == simDim) in cell for
         *  J_x, J_y, J_z
         */
        template<uint32_t T_simDim>
        struct FieldPosition<fields::cellType::Yee, FieldJ, T_simDim>
            : public FieldPosition<fields::cellType::Yee, FieldE, T_simDim>
        {
            HDINLINE FieldPosition()
            {
            }
        };

        /** position (floatD_X in case of T_simDim == simDim) in cell, wrapped in
         * one-component vector since it's a scalar field with only one component, for the
         * scalar field FieldTmp
         */
        template<uint32_t T_simDim>
        struct FieldPosition<fields::cellType::Yee, FieldTmp, T_simDim>
        {
            using FieldPos = pmacc::math::Vector<float_X, T_simDim>;
            using ReturnType = pmacc::math::Vector<FieldPos, DIM1>;

            /// boost::result_of hints
            template<class>
            struct result;

            template<class F>
            struct result<F()>
            {
                using type = ReturnType;
            };

            HDINLINE FieldPosition()
            {
            }

            HDINLINE ReturnType operator()() const
            {
                return ReturnType(FieldPos::create(0.0));
            }
        };

    } // namespace traits
} // namespace picongpu
