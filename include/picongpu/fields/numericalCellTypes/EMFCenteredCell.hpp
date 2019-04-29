/* Copyright 2013-2019 Heiko Burau, Rene Widera, Axel Huebl
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
namespace numericalCellTypes
{

    struct EMFCenteredCell{};

} //namespace numericalCellTypes

namespace traits
{
    /** position (floatD_X in case of T_simDim == simDim) in cell for
     *  E_x, E_y, E_z
     */
    template<uint32_t T_simDim>
    struct FieldPosition<numericalCellTypes::EMFCenteredCell, FieldE, T_simDim>
    {
        typedef pmacc::math::Vector<float_X, T_simDim> PosType;
        typedef const pmacc::math::Vector<PosType, DIM3> ReturnType;

        /// boost::result_of hints
        template<class> struct result;

        template<class F>
        struct result<F()> {
            using type = ReturnType;
        };

        HDINLINE FieldPosition()
        {
        }

        HDINLINE ReturnType operator()() const
        {
            const auto center = PosType::create( 0.5 );

            return ReturnType::create( center );
        }
    };

    /** position (floatD_X in case of T_simDim == simDim) in cell for
     *  B_x, B_y, B_z
     */
    template<uint32_t T_simDim>
    struct FieldPosition<numericalCellTypes::EMFCenteredCell, FieldB, T_simDim> :
        public FieldPosition<numericalCellTypes::EMFCenteredCell, FieldE, T_simDim>
    {
        HDINLINE FieldPosition()
        {
        }
    };

    /** position (float2_X) in cell for J_x, J_y, J_z */
    template<>
    struct FieldPosition<numericalCellTypes::EMFCenteredCell, FieldJ, DIM2>
    {
        /** \tparam float2_X position of the component in the cell
         *  \tparam DIM3     Fields (E/B/J) have 3 components, even in 1 or 2D !
         */
        using VectorVector2D3V = const ::pmacc::math::Vector<
           float2_X,
           DIM3
        >;
        /// boost::result_of hints
        template<class> struct result;

        template<class F>
        struct result<F()> {
            using type = VectorVector2D3V;
        };

        HDINLINE FieldPosition()
        {
        }

        HDINLINE VectorVector2D3V operator()() const
        {
            const float2_X posJ_x( 0.5, 0.0 );
            const float2_X posJ_y( 0.0, 0.5 );
            const float2_X posJ_z( 0.0, 0.0 );

            return VectorVector2D3V( posJ_x, posJ_y, posJ_z );
        }
    };

    /** position (float3_X) in cell for J_x, J_y, J_z
     */
    template<>
    struct FieldPosition<numericalCellTypes::EMFCenteredCell, FieldJ, DIM3>
    {
        /** \tparam float2_X position of the component in the cell
         *  \tparam DIM3     Fields (E/B/J) have 3 components, even in 1 or 2D !
         */
        using VectorVector3D3V = const ::pmacc::math::Vector<
            float3_X,
            DIM3
        >;
        /// boost::result_of hints
        template<class> struct result;

        template<class F>
        struct result<F()> {
            using type = VectorVector3D3V;
        };

        HDINLINE FieldPosition()
        {
        }

        HDINLINE VectorVector3D3V operator()() const
        {
            const float3_X posJ_x( 0.5, 0.0, 0.0 );
            const float3_X posJ_y( 0.0, 0.5, 0.0 );
            const float3_X posJ_z( 0.0, 0.0, 0.5 );

            return VectorVector3D3V( posJ_x, posJ_y, posJ_z );
        }
    };

    /** position (floatD_X in case of T_simDim == simDim) in cell, wrapped in
     * one-component vector since it's a scalar field with only one component, for the
     * scalar field FieldTmp
     */
    template<uint32_t T_simDim>
    struct FieldPosition<numericalCellTypes::EMFCenteredCell, FieldTmp, T_simDim>
    {
        typedef pmacc::math::Vector<float_X, T_simDim> FieldPos;
        typedef pmacc::math::Vector<FieldPos, DIM1> ReturnType;

        /// boost::result_of hints
        template<class> struct result;

        template<class F>
        struct result<F()> {
            using type = ReturnType;
        };

        HDINLINE FieldPosition()
        {
        }

        HDINLINE ReturnType operator()() const
        {
            return ReturnType( FieldPos::create(0.0) );
        }
    };
} // namespace traits
} // picongpu
