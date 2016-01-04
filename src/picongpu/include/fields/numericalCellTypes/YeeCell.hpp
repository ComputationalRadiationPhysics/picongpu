/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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

#include "simulation_defines.hpp"
#include "fields/Fields.def"
#include "math/Vector.hpp"

namespace picongpu
{
namespace yeeCell
{
namespace traits
{
    /** \tparam floatD_X position of the component in the cell
     *  \tparam DIM3     Fields (E/B/J) have 3 components, even in 1 or 2D ! */
    //typedef ::PMacc::math::Vector<floatD_X,DIM3> VectorVector;
    typedef const ::PMacc::math::Vector<float2_X,DIM3> VectorVector2D3V;
    typedef const ::PMacc::math::Vector<float3_X,DIM3> VectorVector3D3V;

    template<typename T_Field, uint32_t T_simDim = simDim>
    struct FieldPosition;

    /** position (float2_X) in cell for E_x, E_y, E_z */
    template<>
    struct FieldPosition<FieldE, DIM2>
    {
        /* boost::result_of hints */
        template<class> struct result;

        template<class F>
        struct result<F()> {
            typedef VectorVector2D3V type;
        };

        HDINLINE VectorVector2D3V operator()() const
        {
            const float2_X posE_x(0.5, 0.0);
            const float2_X posE_y(0.0, 0.5);
            const float2_X posE_z(0.0, 0.0);

            return VectorVector2D3V(posE_x, posE_y, posE_z);
        }
    };

    /** position (float3_X) in cell for E_x, E_y, E_z */
    template<>
    struct FieldPosition<FieldE, DIM3>
    {
        /* boost::result_of hints */
        template<class> struct result;

        template<class F>
        struct result<F()> {
            typedef VectorVector3D3V type;
        };

        HDINLINE VectorVector3D3V operator()() const
        {
            const float3_X posE_x(0.5, 0.0, 0.0);
            const float3_X posE_y(0.0, 0.5, 0.0);
            const float3_X posE_z(0.0, 0.0, 0.5);

            return VectorVector3D3V(posE_x, posE_y, posE_z);
        }
    };

    /** position (float2_X) in cell for B_x, B_y, B_z */
    template<>
    struct FieldPosition<FieldB, DIM2>
    {
        /* boost::result_of hints */
        template<class> struct result;

        template<class F>
        struct result<F()> {
            typedef VectorVector2D3V type;
        };

        HDINLINE VectorVector2D3V operator()() const
        {
            const float2_X posB_x(0.0, 0.5);
            const float2_X posB_y(0.5, 0.0);
            const float2_X posB_z(0.5, 0.5);

            return VectorVector2D3V(posB_x, posB_y, posB_z);
        }
    };

    /** position (float3_X) in cell for B_x, B_y, B_z */
    template<>
    struct FieldPosition<FieldB, DIM3>
    {
        /* boost::result_of hints */
        template<class> struct result;

        template<class F>
        struct result<F()> {
            typedef VectorVector3D3V type;
        };

        HDINLINE VectorVector3D3V operator()() const
        {
            const float3_X posB_x(0.0, 0.5, 0.5);
            const float3_X posB_y(0.5, 0.0, 0.5);
            const float3_X posB_z(0.5, 0.5, 0.0);

            return VectorVector3D3V(posB_x, posB_y, posB_z);
        }
    };

    /** position (float2_X) in cell for J_x, J_y, J_z */
    template<>
    struct FieldPosition<FieldJ, DIM2>
    {
        /* boost::result_of hints */
        template<class> struct result;

        template<class F>
        struct result<F()> {
            typedef VectorVector2D3V type;
        };

        HDINLINE VectorVector2D3V operator()() const
        {
            const FieldPosition<FieldE, DIM2> fieldPosE;
            return fieldPosE();
        }
    };

    /** position (float3_X) in cell for J_x, J_y, J_z */
    template<>
    struct FieldPosition<FieldJ, DIM3>
    {
        /* boost::result_of hints */
        template<class> struct result;

        template<class F>
        struct result<F()> {
            typedef VectorVector3D3V type;
        };

        HDINLINE VectorVector3D3V operator()() const
        {
            const FieldPosition<FieldE, DIM3> fieldPosE;
            return fieldPosE();
        }
    };

    /** position (float2_X) in cell for the scalar field FieldTmp */
    template<>
    struct FieldPosition<FieldTmp, DIM2>
    {
        /* boost::result_of hints */
        template<class> struct result;

        template<class F>
        struct result<F()> {
            typedef float2_X type;
        };

        HDINLINE float2_X operator()() const
        {
            return float2_X(0.0, 0.0);
        }
    };

    /** position (float3_X) in cell for the scalar field FieldTmp */
    template<>
    struct FieldPosition<FieldTmp, DIM3>
    {
        /* boost::result_of hints */
        template<class> struct result;

        template<class F>
        struct result<F()> {
            typedef float3_X type;
        };

        HDINLINE float3_X operator()() const
        {
            return float3_X(0.0, 0.0, 0.0);
        }
    };
} // traits
} // yeeCell
} // picongpu
