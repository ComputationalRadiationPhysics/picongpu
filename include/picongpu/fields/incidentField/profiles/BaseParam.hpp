/* Copyright 2023 Sergei Bastrakov, Finn-Ole Carstens
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/incidentField/profiles/BaseParam.def"

#include <pmacc/algorithms/math/defines/pi.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace profiles
            {
                namespace detail
                {
                    /** Internal version of the base parameters structure
                     *
                     * @tparam T_BaseParam parameter structure matching BaseParam requirements
                     */
                    template<typename T_BaseParam>
                    struct BaseParamUnitless : public T_BaseParam
                    {
                        //! User SI parameters
                        using Params = T_BaseParam;

                        /** Wave length along propagation direction
                         *
                         * unit: sim.unit.length()
                         */
                        static constexpr float_X WAVE_LENGTH
                            = static_cast<float_X>(Params::WAVE_LENGTH_SI / sim.unit.length());

                        /** Frequency
                         *
                         * unit: 1/sim.unit.time()
                         */
                        static constexpr float_X f = static_cast<float_X>(sim.pic.getSpeedOfLight() / WAVE_LENGTH);

                        /** Angular frequency
                         *
                         * unit: 1/sim.unit.time()
                         */
                        static constexpr float_X w = pmacc::math::Pi<float_X>::doubleValue * f;

                        /** Max amplitude of E field
                         *
                         * unit: sim.unit.eField()
                         */
                        static constexpr float_X AMPLITUDE
                            = static_cast<float_X>(Params::AMPLITUDE_SI / sim.unit.eField());

                        /** Pulse duration
                         *
                         * unit: sim.unit.time()
                         */
                        static constexpr float_X PULSE_DURATION
                            = static_cast<float_X>(Params::PULSE_DURATION_SI / sim.unit.time());

                        // Some utility that is not part of public interface
                    private:
                        //! Return z direction in 3d or 0 in 2d
                        HDINLINE static constexpr float_64 getDirectionZ()
                        {
                            if constexpr(simDim == 3)
                                return Params::DIRECTION_Z;
                            return 0.0;
                        }

                        //! Return z focus position in 3d or 0 in 2d
                        HINLINE static constexpr float_64 getFocusPositionZ()
                        {
                            if constexpr(simDim == 3)
                                return Params::FOCUS_POSITION_Z_SI;
                            return 0.0;
                        }

                        /** SFINAE deduction if the user parameter define the variable TIME_DELAY_SI
                         *
                         * This allows that time delay can be an optional variable a user must only define if needed.
                         * The default if it is not defined is 0.
                         * @{
                         */
                        template<typename T, typename = void>
                        struct GetTimeDelay
                        {
                            static constexpr float_X value = 0.0;
                        };

                        template<typename T>
                        struct GetTimeDelay<T, decltype((void) T::TIME_DELAY_SI, void())>
                        {
                            static constexpr float_X value = T::TIME_DELAY_SI;
                        };

                    public:
                        /** Time delay
                         *
                         * unit: sim.unit.time()
                         */
                        static constexpr float_X TIME_DELAY
                            = static_cast<float_X>(GetTimeDelay<Params>::value / sim.unit.time());
                        PMACC_CASSERT_MSG(
                            _error_laser_time_delay_must_be_positive____check_your_incidentField_param_file,
                            (TIME_DELAY >= 0.0));

                        /** Unit propagation direction vector in 3d
                         *
                         * In 2d simulations, z component is always set to 0.
                         *
                         * @{
                         */
                        static constexpr float_X DIR_X = static_cast<float_X>(Params::DIRECTION_X);
                        static constexpr float_X DIR_Y = static_cast<float_X>(Params::DIRECTION_Y);
                        static constexpr float_X DIR_Z = static_cast<float_X>(getDirectionZ());
                        /** @} */

                        // Check that direction is normalized
                        static constexpr float_64 dirNorm2 = DIR_X * DIR_X + DIR_Y * DIR_Y + DIR_Z * DIR_Z;
                        PMACC_CASSERT_MSG(
                            _error_laser_direction_vector_must_be_unit____check_your_incidentField_param_file,
                            (dirNorm2 > 0.9999) && (dirNorm2 < 1.0001));

                        /** Unit polarization direction vector
                         *
                         * @{
                         */
                        static constexpr float_X POL_DIR_X = static_cast<float_X>(Params::POLARISATION_DIRECTION_X);
                        static constexpr float_X POL_DIR_Y = static_cast<float_X>(Params::POLARISATION_DIRECTION_Y);
                        static constexpr float_X POL_DIR_Z = static_cast<float_X>(Params::POLARISATION_DIRECTION_Z);
                        /** @} */

                        // Check that polarization direction is normalized
                        static constexpr float_64 polDirNorm2
                            = POL_DIR_X * POL_DIR_X + POL_DIR_Y * POL_DIR_Y + POL_DIR_Z * POL_DIR_Z;
                        PMACC_CASSERT_MSG(
                            _error_laser_polarization_direction_vector_must_be_unit____check_your_incidentField_param_file,
                            (polDirNorm2 > 0.9999) && (polDirNorm2 < 1.0001));

                        // Check that polarization direction is orthogonal to propagation direction
                        static constexpr float_64 dotPropagationPolarization
                            = DIR_X * POL_DIR_X + DIR_Y * POL_DIR_Y + DIR_Z * POL_DIR_Z;
                        PMACC_CASSERT_MSG(
                            _error_laser_polarization_direction_vector_must_be_orthogonal_to_propagation_direction____check_your_incidentField_param_file,
                            (dotPropagationPolarization > -0.0001) && (dotPropagationPolarization < 0.0001));

                        /** Focus position in total cooridnate system
                         *
                         * unit: sim.unit.length()
                         *
                         * @{
                         */
                        static constexpr float_X FOCUS_POSITION_X
                            = static_cast<float_X>(Params::FOCUS_POSITION_X_SI / sim.unit.length());
                        static constexpr float_X FOCUS_POSITION_Y
                            = static_cast<float_X>(Params::FOCUS_POSITION_Y_SI / sim.unit.length());
                        static constexpr float_X FOCUS_POSITION_Z
                            = static_cast<float_X>(getFocusPositionZ() / sim.unit.length());
                        /** @} */
                    };

                    /** Base parameter structure for separable lasers with transversal Gaussian profile
                     *
                     * Constitutes base parameter structure extended with waist size parameters.
                     *
                     * @tparam T_BaseParam parameter structure matching BaseParam requirements
                     *                     must define W0_AXIS_1_SI, W0_AXIS_2_SI
                     */
                    template<typename T_BaseParam>
                    struct BaseTransversalGaussianParamUnitless : public BaseParamUnitless<T_BaseParam>
                    {
                        //! User SI parameters
                        using Params = T_BaseParam;

                        // unit: sim.unit.length()
                        static constexpr float_X W0_AXIS_1
                            = static_cast<float_X>(Params::W0_AXIS_1_SI / sim.unit.length());
                        // unit: sim.unit.length()
                        static constexpr float_X W0_AXIS_2
                            = static_cast<float_X>(Params::W0_AXIS_2_SI / sim.unit.length());
                    };

                } // namespace detail
            } // namespace profiles
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
