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

#include "picongpu/fields/incidentField/profiles/BaseParam.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>


namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            //! Concept defining interface of incident field functors for E and B
            struct FunctorIncidentFieldConcept
            {
                /** Create a functor on the host side for the given time step
                 *
                 * Since it is host-only, one could access global simulation data like Environment<simDim> and objects
                 * controlled by DataConnector. Note that it is not possible in operator(), but relevant data can be
                 * saved in members on this class.
                 * Time-dependent part of the functor can also be precalculated on host and saved in members.
                 *
                 * @param currentStep current time step index, note that it is fractional
                 * @param unitField conversion factor from SI to internal units,
                 *                  field_internal = field_SI / unitField
                 */
                HINLINE FunctorIncidentFieldConcept(float_X currentStep, float3_64 unitField);

                //! Functor must be copiable to device by value
                // HDINLINE FunctorIncidentFieldConcept(const FunctorIncidentFieldConcept& other) = default;

                /** Return incident field for the given position
                 *
                 * Note that component matching the source boundary (e.g. the y component for YMin and YMax boundaries)
                 * will not be used, and so should be zero.
                 *
                 * @param totalCellIdx cell index in the total domain (including all moving window slides),
                 *        note that it is fractional
                 * @return incident field value in internal units
                 */
                HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const;
            };

            //! Helper incident field functor always returning 0
            struct ZeroFunctor
            {
                /** Create a functor on the host side for the given time step
                 *
                 * @param currentStep current time step index, note that it is fractional
                 * @param unitField conversion factor from SI to internal units,
                 *                  field_internal = field_SI / unitField
                 */
                HINLINE ZeroFunctor(float_X const currentStep, float3_64 const unitField)
                {
                }

                /** Return zero incident field for any given position
                 *
                 * @param totalCellIdx cell index in the total domain (including all moving window slides),
                 *        note that it is fractional
                 * @return incident field value in internal units
                 */
                HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
                {
                    return float3_X::create(0.0_X);
                }
            };
            namespace detail
            {
                /** Base class for calculating incident E functors
                 *
                 * Defines internal coordinate system tied to laser focus position and direction.
                 * Its axes are, in this order: propagation direction, polarization direction,
                 * cross product of propagation direction and polarization direction.
                 *
                 * Provides conversion operations for cooridnate and time transforms to internal system.
                 * Essentially, a client of this class can implement a laser in internal coordinate system and
                 * use the base class functionality for all necessary transformations.
                 *
                 * Checks unit matching.
                 *
                 * @tparam T_BaseParam parameter structure matching profiles::BaseParam requirements
                 */
                template<typename T_BaseParam>
                class BaseFunctorE
                {
                public:
                    //! Unitless base parameters type
                    using Unitless = profiles::detail::BaseParamUnitless<T_BaseParam>;

                    /** Create a functor on the host side, check that unit matches the internal E unit
                     *
                     * @param currentStep current time step index, note that it is fractional
                     * @param unitField conversion factor from SI to internal units,
                     *                  fieldE_internal = fieldE_SI / unitField
                     */
                    HINLINE BaseFunctorE(float_X const currentStep, float3_64 const unitField)
                        : currentTimeOrigin(currentStep * DELTA_T)
                        , origin(getOrigin())
                    {
                        checkUnit(unitField);
                    }

                    //! Get a normalized 3-dimensional direction vector
                    HDINLINE static float3_X getDirection()
                    {
                        return getAxis0();
                    }

                    /** Get current time to calculate field at the given point
                     *
                     * It accounts for both current PIC iteration and location relative to origin.
                     * Note that the result may be negative as well, and clients may want to set
                     * field = 0 when the returned value is negative.
                     *
                     * @param totalCellIdx cell index in the total domain
                     * @param phaseVelocity phase velocity along the propagation direction
                     */
                    HDINLINE float_X
                    getCurrentTime(floatD_X const& totalCellIdx, float_X const phaseVelocity = SPEED_OF_LIGHT) const
                    {
                        auto const shiftFromOrigin = totalCellIdx * cellSize.shrink<simDim>() - origin;
                        auto const distance = pmacc::math::dot(shiftFromOrigin, getDirection().shrink<simDim>());
                        auto const timeDelay = distance / phaseVelocity;
                        return currentTimeOrigin - timeDelay;
                    }

                    //! Get a unit vector with linear E polarization
                    HDINLINE float3_X getLinearPolarizationVector() const
                    {
                        return getAxis1();
                    }

                    //! Get a first vector with circular E polarization, norm is 0.5
                    HDINLINE float3_X getCircularPolarizationVector1() const
                    {
                        return getLinearPolarizationVector() / math::sqrt(2.0_X);
                    }

                    //! Get a second vector with circular E polarization, norm is 0.5
                    HDINLINE float3_X getCircularPolarizationVector2() const
                    {
                        return pmacc::math::cross(getAxis0(), getCircularPolarizationVector1());
                    }

                    /** Transform the given cell index to coordinates (not cell index) in the internal system
                     *
                     * @param totalCellIdx cell index in the total domain
                     */
                    HDINLINE floatD_X getInternalCoordinates(floatD_X const& totalCellIdx) const
                    {
                        auto const shiftFromOrigin = totalCellIdx * cellSize.shrink<simDim>() - origin;
                        floatD_X result;
                        result[0] = pmacc::math::dot(shiftFromOrigin, getAxis0().shrink<simDim>());
                        result[1] = pmacc::math::dot(shiftFromOrigin, getAxis1().shrink<simDim>());
                        if constexpr(simDim == 3)
                            result[2] = pmacc::math::dot(shiftFromOrigin, getAxis2().shrink<simDim>());
                        return result;
                    }

                protected:
                    /** Laser center at generation surface when projected along propagation direction
                     *
                     * That point serves as origin in internal coordinate system
                     */
                    floatD_X const origin;

                    /** Current time for calculating the field at the origin
                     *
                     * Other points will have time shifts relative to it according to their position relative to the
                     * origin and propagation direction.
                     */
                    float_X const currentTimeOrigin;

                    //! Calculate origin position
                    HINLINE static floatD_X getOrigin()
                    {
                        /* Find min value of variable p so that a line
                         * line(p) = focusPosition + p * direction
                         * intersects with the Huygens surface.
                         * That intersection point is where the laser is centered when entering the volume.
                         * Use that as origin of the internal coordinate system.
                         * Note that it's generally not the first point of entry, as that is one of vertices.
                         */
                        auto const& subGrid = Environment<simDim>::get().SubGrid();
                        auto const totalDomainCells = subGrid.getTotalDomain().size;
                        auto const direction = getDirection().shrink<simDim>();
                        auto const focus = float3_X(
                                               Unitless::FOCUS_POSITION_X,
                                               Unitless::FOCUS_POSITION_Y,
                                               Unitless::FOCUS_POSITION_Z)
                                               .shrink<simDim>();
                        auto firstIntersectionP = std::numeric_limits<float_X>::infinity();
                        for(uint32_t axis = 0u; axis < simDim; ++axis)
                        {
                            // The expressions should generally work anyways, but to avoid potential 0/0
                            if(std::abs(direction[axis]) > std::numeric_limits<float_X>::epsilon())
                            {
                                // Take into account 0.75 cells inwards shift of Huygens surface
                                auto const minPosition
                                    = (static_cast<float_X>(OFFSET[axis][0]) + 0.75_X) * cellSize[axis];
                                firstIntersectionP
                                    = std::min(firstIntersectionP, (minPosition - focus[axis]) / direction[axis]);
                                auto const maxPosition
                                    = (static_cast<float_X>(totalDomainCells[axis] - OFFSET[axis][1]) - 0.75_X)
                                    * cellSize[axis];
                                firstIntersectionP
                                    = std::min(firstIntersectionP, (maxPosition - focus[axis]) / direction[axis]);
                            }
                        }
                        return focus + firstIntersectionP * direction;
                    }

                    /** Get unit axis vectors of internal cooridnate system
                     *
                     * For simplicity of use and since they are for internal use only, these are always in 3d.
                     *
                     * @{
                     */
                    HDINLINE static constexpr float3_X getAxis0()
                    {
                        return float3_X(Unitless::DIR_X, Unitless::DIR_Y, Unitless::DIR_Z);
                    }

                    HDINLINE static constexpr float3_X getAxis1()
                    {
                        return float3_X(Unitless::POL_DIR_X, Unitless::POL_DIR_Y, Unitless::POL_DIR_Z);
                    }

                    HDINLINE static constexpr float3_X getAxis2()
                    {
                        // cross product of getAxis0() and getAxis1()
                        return float3_X(
                            Unitless::DIR_Y * Unitless::POL_DIR_Z - Unitless::DIR_Z * Unitless::POL_DIR_Y,
                            Unitless::DIR_Z * Unitless::POL_DIR_X - Unitless::DIR_X * Unitless::POL_DIR_Z,
                            Unitless::DIR_X * Unitless::POL_DIR_Y - Unitless::DIR_Y * Unitless::POL_DIR_X

                        );
                    }
                    /** @} */

                    //! Check that the input units are valid
                    HINLINE static void checkUnit(float3_64 const unitField)
                    {
                        // Ensure that we always get unitField = (UNIT_EFIELD, UNIT_EFIELD, UNIT_EFIELD) so that
                        // we can always calculate in internal units and avoid conversions in child types.
                        // We can afford it each time, as this is done on host before kernel
                        for(uint32_t axis = 0; axis < 3; axis++)
                        {
                            constexpr double ulp = 1.0;
                            constexpr double eps = std::numeric_limits<double>::epsilon();
                            bool const isMatchingUnit = (std::fabs(unitField[axis] - UNIT_EFIELD) <= eps * ulp);
                            if(!isMatchingUnit)
                                throw std::runtime_error(
                                    "Incident field BaseFunctorE created with wrong unit: expected "
                                    + std::to_string(UNIT_EFIELD) + ", got " + std::to_string(unitField[axis]));
                        }
                    }
                };

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

                    /** Create a functor on the host side for the given time step
                     *
                     * @param currentStep current time step index, note that it is fractional
                     * @param unitField conversion factor from SI to internal units,
                     *                  fieldB_internal = fieldB_SI / unitField
                     */
                    HINLINE ApproximateIncidentB(float_X const currentStep, float3_64 const unitField)
                        : Base(currentStep, unitField * unitConversionBtoE)
                    {
                    }

                    /** Calculate B value using SVEA
                     *
                     * The resulting value is calculated as B = cross(k, E) / c, where
                     * k is pulse propagation direction vector defined by T_axis, T_direction
                     * E is value returned by a base functor at the target location and time of resulting B
                     *
                     * @param totalCellIdx cell index in the total domain (including all moving window slides)
                     * @return incident field B value in internal units
                     */
                    HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
                    {
                        // Get corresponding E value, it is already in internal units
                        auto const eValue = Base::operator()(totalCellIdx);
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
