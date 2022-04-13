/* Copyright 2020-2022 Sergei Bastrakov
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

#include <cstdint>

namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            //! Concept defining interface of incident field functors for E and B
            struct FunctorIncidentFieldConcept
            {
                /** Create a functor on the host side
                 *
                 * Since it is host-only, one could access global simulation data like Environment<simDim> and objects
                 * controlled by DataConnector. Note that it is not possible in operator(), but relevant data can be
                 * saved in members on this class.
                 *
                 * @param unitField conversion factor from SI to internal units,
                 *                  field_internal = field_SI / unitField
                 */
                HINLINE FunctorIncidentFieldConcept(float3_64 unitField);

                //! Functor must be copiable to device by value
                // HDINLINE FunctorIncidentFieldConcept(const FunctorIncidentFieldConcept& other) = default;

                /** Return incident field for the given position and time.
                 *
                 * Note that component matching the source boundary (e.g. the y component for YMin and YMax boundaries)
                 * will not be used, and so should be zero.
                 *
                 * @param totalCellIdx cell index in the total domain (including all moving window slides),
                 *        note that it is fractional
                 * @param currentStep current time step index, note that it is fractional
                 * @return incident field value in internal units
                 */
                HDINLINE float3_X operator()(floatD_X const& totalCellIdx, float_X currentStep) const;
            };

            //! Helper incident field functor always returning 0
            struct ZeroFunctor
            {
                /** Create a functor on the host side
                 *
                 * @param unitField conversion factor from SI to internal units,
                 *                  field_internal = field_SI / unitField
                 */
                HINLINE ZeroFunctor(float3_64 unitField)
                {
                }

                /** Return zero incident field for any given position and time.
                 *
                 * @param totalCellIdx cell index in the total domain (including all moving window slides),
                 *        note that it is fractional
                 * @param currentStep current time step index, note that it is fractional
                 * @return incident field value in internal units
                 */
                HDINLINE float3_X operator()(floatD_X const& totalCellIdx, float_X currentStep) const
                {
                    return float3_X::create(0.0_X);
                }
            };
            namespace detail
            {
                /** Helper functor to calculate values of B from values of E using slowly varying envelope
                 * approximation (SVEA) for the given axis and direction
                 *
                 * The functor follows FunctorIncidentFieldConcept and thus can be used as FunctorIncidentB.
                 *
                 * @tparam T_FunctorIncidentE functor for the incident E field, follows the interface of
                 *                            FunctorIncidentFieldConcept (defined in Functors.hpp),
                 *                            must have been applied for the same axis and direction
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 */
                template<typename T_FunctorIncidentE, uint32_t T_axis, int32_t T_direction>
                class ApproximateIncidentB : public T_FunctorIncidentE
                {
                public:
                    //! Base class
                    using Base = T_FunctorIncidentE;

                    //! Relation between unitField for E and B: E = B * unitConversionBtoE
                    static constexpr float_64 unitConversionBtoE = UNIT_EFIELD / UNIT_BFIELD;

                    /** Create a functor on the host side
                     *
                     * @param unitField conversion factor from SI to internal units,
                     *                  fieldB_internal = fieldB_SI / unitField
                     */
                    HINLINE ApproximateIncidentB(const float3_64 unitField) : Base(unitField * unitConversionBtoE)
                    {
                    }

                    /** Calculate B value using SVEA
                     *
                     * The resulting value is calculated as B = cross(k, E) / c, where
                     * k is pulse propagation direction vector defined by T_axis, T_direction
                     * E is value returned by a base functor at the target location and time of resulting B
                     *
                     * @param totalCellIdx cell index in the total domain (including all moving window slides)
                     * @param currentStep current time step index, note that it is fractional
                     * @return incident field B value in internal units
                     */
                    HDINLINE float3_X operator()(const floatD_X& totalCellIdx, const float_X currentStep) const
                    {
                        // Get corresponding E value, it is already in internal units
                        auto const eValue = Base::operator()(totalCellIdx, currentStep);
                        // To avoid making awkward type casts and calling cross product, we express it manually as
                        // rotation and sign change
                        constexpr float_X signAndNormalization = static_cast<float_X>(T_direction) / SPEED_OF_LIGHT;
                        constexpr uint32_t dir0 = T_axis;
                        constexpr uint32_t dir1 = (dir0 + 1) % 3;
                        constexpr uint32_t dir2 = (dir0 + 2) % 3;
                        auto bValue = float3_X::create(0.0_X);
                        bValue[dir1] = -eValue[dir2] * signAndNormalization;
                        bValue[dir2] = eValue[dir1] * signAndNormalization;
                        return bValue;
                    }
                };
            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
