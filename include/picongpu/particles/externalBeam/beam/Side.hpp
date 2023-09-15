/* Copyright 2020-2023 Pawel Ordyna
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

#include <pmacc/math/vector/TwistComponents.hpp>

#include <type_traits>

#include <pmacc/math/vector/compile-time/TwistComponents.hpp>

namespace picongpu::particles::externalBeam::beam
{
    /* This file defines the possible base beam orientations.
     *
     *  Example: X Side
     *      The beam propagates along the x axis ( PIC coordinate system).
     *      The base position of the beam coordinate system (0,0,0)  is placed
     *      in the middle of the x_PIC=0 plane.
     *      That is at (0, 0.5 * Y, 0.5 * Z), where Y and Z are the lengths of
     *      the simulation box sides along y_PIC and z_PIC axes.
     *      Therefore BeamStartPosition = ( 0.0, 0.5, 0.5 ) for the XSide.
     *
     *      SimAxesInBeamOrder  defines how the 3 directions
     *      (x, y, z) in the PIC system correspond to the ones in the beam
     *      system. The true/false in reverse define the relative orientations. For XSide:
     *      SimAxesInBeamOrder = ( 2, 1, 0) and reverse_beam_x = true, reverse_beam_y = false, reverse_beam_z = false
     *      says:
     *          * x_beam = - z_PIC,
     *          * y_beam = y_PIC,
     *          * z_beam = x_PIC
     */

    // Start Positions for all orientations:
    struct XStartPosition
    {
        static constexpr float_X x{0.0};
        static constexpr float_X y{0.5};
        static constexpr float_X z{0.5};
    };
    struct XRStartPosition
    {
        static constexpr float_X x{1.0};
        static constexpr float_X y{0.5};
        static constexpr float_X z{0.5};
    };
    struct YStartPosition
    {
        static constexpr float_X x{0.5};
        static constexpr float_X y{0.0};
        static constexpr float_X z{0.5};
    };
    struct YRStartPosition
    {
        static constexpr float_X x{0.5};
        static constexpr float_X y{1.0};
        static constexpr float_X z{0.5};
    };
    struct ZStartPosition
    {
        static constexpr float_X x{0.5};
        static constexpr float_X y{0.5};
        static constexpr float_X z{0.0};
    };
    struct ZRStartPosition
    {
        static constexpr float_X x{0.5};
        static constexpr float_X y{0.5};
        static constexpr float_X z{1.0};
    };

    /** Defines the probing coordinate system
     *
     * @tparam T_BeamStartPosition origin of the beam system ([0,1) relative position in the total volume)
     * @tparam T_SimAxesInBeamOrder defines how the 3 directions  (x, y, z) in the PIC system correspond
     *      to the ones in the beam.
     * @tparam reverse_beam_x beam x coordinate orientation
     * @tparam reverse_beam_y beam x coordinate orientation
     * @tparam reverse_beam_z beam x coordinate orientation
     */
    template<
        typename T_BeamStartPosition,
        typename T_SimAxesInBeamOrder,
        bool reverse_beam_x,
        bool reverse_beam_y,
        bool reverse_beam_z>
    struct ProbingCoordinates
    {
        static constexpr float_X beamStartPosition[3]{
            T_BeamStartPosition::x,
            T_BeamStartPosition::y,
            T_BeamStartPosition::z};
        using SimAxesInBeamOrder = T_SimAxesInBeamOrder;

        using BeamAxesInSimOrder = typename pmacc::math::CT::Int<
            pmacc::mp_find<SimAxesInBeamOrder, std::integral_constant<int, 0>>::value,
            pmacc::mp_find<SimAxesInBeamOrder, std::integral_constant<int, 1>>::value,
            pmacc::mp_find<SimAxesInBeamOrder, std::integral_constant<int, 2>>::value>;

        // Compile time axes order conversion
        template<typename T_Vector>
        using TwistSimToBeam_t = typename pmacc::math::CT::TwistComponents<T_Vector, SimAxesInBeamOrder>::type;
        template<typename T_Vector>
        using TwistBeamToSim_t = typename pmacc::math::CT::TwistComponents<T_Vector, BeamAxesInSimOrder>::type;
        static constexpr bool reverse[3] = {reverse_beam_x, reverse_beam_y, reverse_beam_z};

        // Axis index conversion
        template<uint32_t idx>
        using BeamToSimIdx_t = pmacc::mp_at_c<SimAxesInBeamOrder, idx>;
        template<uint32_t idx>
        using SimToBeamIdx_t = pmacc::mp_at_c<BeamAxesInSimOrder, idx>;

        //! Twist coordinates from sim to beam order for run time vectors (does not change data in memory)
        template<typename T_Vector>
        HDINLINE static auto twistSimToBeam(T_Vector& vector)
        {
            return pmacc::math::twistComponents<SimAxesInBeamOrder>(vector);
        }
        //! Twist coordinates from beam to sim order for run time vectors (does not change memory layout)
        template<typename T_Vector>
        HDINLINE static auto twistBeamToSim(T_Vector& vector)
        {
            return pmacc::math::twistComponents<BeamAxesInSimOrder>(vector);
        }

        //! Rotate vector from sim to beam  system (includes orientation and not is not only twisting coordinates)
        template<typename T_Vector>
        HDINLINE static auto rotateSimToBeam(T_Vector const& vector)
        {
            T_Vector result = T_Vector::create(0.0);
            auto twisted = twistSimToBeam(vector);
            using ComponentType = typename T_Vector::type;
            if constexpr(reverse_beam_x)
                result[0] = static_cast<ComponentType>(-1.0) * twisted[0];
            else
                result[0] = twisted[0];
            if constexpr(reverse_beam_y)
                result[1] = static_cast<ComponentType>(-1.0) * twisted[1];
            else
                result[1] = twisted[1];
            if constexpr(reverse_beam_z)
                result[2] = static_cast<ComponentType>(-1.0) * twisted[2];
            else
                result[2] = twisted[2];
            return result;
        }

        //! Rotate vector from beam to sim system (includes orientation and not is not only twisting coordinates)
        template<typename T_Vector>
        HDINLINE static auto rotateBeamToSim(T_Vector const& vector)
        {
            T_Vector result = T_Vector::create(0.0);
            using ComponentType = typename T_Vector::type;
            if constexpr(reverse_beam_x)
                result[0] = static_cast<ComponentType>(-1.0) * vector[0];
            else
                result[0] = vector[0];
            if constexpr(reverse_beam_y)
                result[1] = static_cast<ComponentType>(-1.0) * vector[1];
            else
                result[1] = vector[1];
            if constexpr(reverse_beam_z)
                result[2] = static_cast<ComponentType>(-1.0) * vector[2];
            else
                result[2] = vector[2];
            return twistBeamToSim(result);
        }

        //! Transform offsets (like a cell index) from sim to beam coordinates
        template<typename T_Vector>
        HDINLINE static T_Vector transformOffsetSimToBeam(T_Vector const& vector, T_Vector const& volumeSize)
        {
            T_Vector result = rotateSimToBeam(vector);
            for(uint32_t i = 1u; i < T_Vector::dim; i++)
                result[i] = math::fmod(vector[i] + volumeSize[i], volumeSize[i]);
            return result;
        }

        //! Transform offsets (like a cell index) from beam to sim coordinates
        template<typename T_Vector>
        HDINLINE static T_Vector transformOffsetBeamToSim(T_Vector const& vector, T_Vector const& volumeSize)
        {
            T_Vector result = rotateBeamToSim(vector);
            for(uint32_t i = 1u; i < T_Vector::dim; i++)
                result[i] = math::fmod(vector[i] + volumeSize[i], volumeSize[i]);
            return result;
        }
    };


    //! Probing along the PIC x basis vector.
    using XSide = ProbingCoordinates<XStartPosition, pmacc::math::CT::Int<2, 1, 0>, true, false, false>;
    //! Probing against the PIC x basis vector.
    using XRSide = ProbingCoordinates<XRStartPosition, pmacc::math::CT::Int<2, 1, 0>, true, true, true>;
    //! Probing along the PIC y basis vector.
    using YSide = ProbingCoordinates<YStartPosition, pmacc::math::CT::Int<2, 0, 1>, true, true, false>;
    //! Probing against the PIC y basis vector.
    using YRSide = ProbingCoordinates<YRStartPosition, pmacc::math::CT::Int<2, 0, 1>, true, false, true>;
    //! Probing along the PIC z basis vector.
    using ZSide = ProbingCoordinates<ZStartPosition, pmacc::math::CT::Int<1, 0, 2>, true, false, false>;
    //! Probing against the PIC z basis vector.
    using ZRSide = ProbingCoordinates<ZRStartPosition, pmacc::math::CT::Int<1, 0, 2>, true, true, true>;
} // namespace picongpu::particles::externalBeam::beam
