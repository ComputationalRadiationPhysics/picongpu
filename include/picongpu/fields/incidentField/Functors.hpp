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

#include "picongpu/fields/incidentField/Traits.hpp"
#include "picongpu/fields/incidentField/profiles/BaseParam.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>


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

                /** Optional interface to get a given field component
                 *
                 * When provided by a functor, this version will be called instead of operator().
                 * Result of functor.getComponent<T_component>(totalCellIdx) must equal
                 * functor(totalCellIdx)[T_Component].
                 *
                 * This interface exists only for performance optimization purposes for (very) compute-intensive
                 * functors. Namely, it allows calculation of only components needed by the incident field solver. Note
                 * however that this is generally not important as incident field is rarely a hot spot.
                 *
                 * @tparam T_component field component, 0 = x, 1 = y, 2 = z
                 *
                 * @param totalCellIdx cell index in the total domain (including all moving window slides),
                 *        note that it is fractional
                 */
                template<uint32_t T_component>
                HDINLINE float_X getComponent(floatD_X const& totalCellIdx) const;
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
                /** SFINAE dedection if the user parameter define the variable FOCUS_ORIGIN_*
                 *
                 * This allows that focus origin can be an optional variable a user must only define if needed.
                 * The default if it is not defined is Origin::Zero
                 * @{
                 */
                template<typename T, typename = void>
                struct GetOriginX
                {
                    static constexpr Origin value = Origin::Zero;
                };

                template<typename T>
                struct GetOriginX<T, decltype((void) T::FOCUS_ORIGIN_X, void())>
                {
                    static constexpr Origin value = T::FOCUS_ORIGIN_X;
                };

                template<typename T, typename = void>
                struct GetOriginY
                {
                    static constexpr Origin value = Origin::Zero;
                };

                template<typename T>
                struct GetOriginY<T, decltype((void) T::FOCUS_ORIGIN_Y, void())>
                {
                    static constexpr Origin value = T::FOCUS_ORIGIN_Y;
                };

                template<typename T, typename = void>
                struct GetOriginZ
                {
                    static constexpr Origin value = Origin::Zero;
                };

                template<typename T>
                struct GetOriginZ<T, decltype((void) T::FOCUS_ORIGIN_Z, void())>
                {
                    static constexpr Origin value = T::FOCUS_ORIGIN_Z;
                };
                /**@}*/

                /** Base class for calculating incident E functors
                 *
                 * Defines internal coordinate system tied to laser focus position and direction.
                 * Its axes are, in this order: propagation direction, polarization direction,
                 * cross product of propagation direction and polarization direction.
                 * The internal coordinate system is always 3d, regardless of simDim.
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
                        : origin(getOrigin())
                        , focus(getFocus())
                        , currentTimeOrigin(currentStep * DELTA_T)
                        , phaseVelocity(getPhaseVelocity())
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
                     *
                     * @{
                     */

                    //! 3d version
                    HDINLINE float_X getCurrentTime(float3_X const& totalCellIdx) const
                    {
                        auto const shiftFromOrigin = totalCellIdx * cellSize - origin;
                        auto const distance = pmacc::math::dot(shiftFromOrigin, getDirection());
                        auto const timeDelay = distance / phaseVelocity;
                        return currentTimeOrigin - timeDelay;
                    }

                    //! 2d version
                    HDINLINE float_X getCurrentTime(float2_X const& totalCellIdx) const
                    {
                        return getCurrentTime(float3_X{totalCellIdx.x(), totalCellIdx.y(), 0.0_X});
                    }

                    /** @} */

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

                    /** Transform the given cell index to 3d coordinates (not cell index) in the internal system
                     *
                     * @param totalCellIdx cell index in the total domain
                     *
                     * @{
                     */

                    //! 3d version
                    HDINLINE float3_X getInternalCoordinates(float3_X const& totalCellIdx) const
                    {
                        auto const shiftFromOrigin = totalCellIdx * cellSize - origin;
                        float3_X result;
                        result[0] = pmacc::math::dot(shiftFromOrigin, getAxis0());
                        result[1] = pmacc::math::dot(shiftFromOrigin, getAxis1());
                        result[2] = pmacc::math::dot(shiftFromOrigin, getAxis2());
                        return result;
                    }

                    //! 2d version, returns a 3d coordinate as the internal coordinate system is always 3d
                    HDINLINE float3_X getInternalCoordinates(float2_X const& totalCellIdx) const
                    {
                        return getInternalCoordinates(float3_X{totalCellIdx.x(), totalCellIdx.y(), 0.0_X});
                    }

                    /** @} */

                protected:
                    /** Laser center at generation surface when projected from focus along (negative) propagation
                     * direction
                     *
                     * That point serves as origin in internal coordinate system.
                     * The laser is transversally centered around the origin.
                     * It is always 3d, z component is set to 0 in 2d.
                     */
                    float3_X const origin;

                    float3_X const focus;

                    /** Current time for calculating the field at the origin
                     *
                     * Other points will have time shifts relative to it according to their position relative to the
                     * origin and propagation direction.
                     */
                    float_X const currentTimeOrigin;

                    /** Phase velocity for the enabled field solver
                     *
                     * The solver-fitting phase velocity ensures proper coupling wrt time delays.
                     */
                    float_X const phaseVelocity;

                    //! Calculate focus position
                    HINLINE static float3_X getFocus()
                    {
                        auto const& subGrid = Environment<simDim>::get().SubGrid();
                        auto const globalDomainCells = subGrid.getGlobalDomain().size;
                        auto result = float3_X(
                            Unitless::FOCUS_POSITION_X,
                            Unitless::FOCUS_POSITION_Y,
                            Unitless::FOCUS_POSITION_Z);

                        if constexpr(GetOriginX<Unitless>::value == Origin::Center)
                        {
                            result.x() += static_cast<float_X>(globalDomainCells.x() / 2u) * cellSize.x();
                        }
                        if constexpr(GetOriginY<Unitless>::value == Origin::Center)
                            result.y() += static_cast<float_X>(globalDomainCells.y() / 2u) * cellSize.y();

                        // if condition is guarded against aout of memory access for 2D simulations
                        if constexpr(GetOriginZ<Unitless>::value == Origin::Center && simDim == DIM3)
                            result.z() += static_cast<float_X>(globalDomainCells.z() / 2u) * cellSize.z();

                        return result;
                    }

                    //! Calculate origin position
                    HINLINE static float3_X getOrigin()
                    {
                        /* The origin is calculated as a projection from the focus position onto the generation surface
                         * along the negative propagation direction.
                         * Thus, this point is an intersection of the Huygens surface with a line
                         * line(p) = focusPosition + p * direction.
                         * Between the (normally, two) intersection points we choose one encountered by a laser first,
                         * so with the smaller of p values in the formula above.
                         * Note that the origin is generally not the first point of entry to or domain, as that would
                         * be one of vertices. However it is a transversal center of the laser at the generation
                         * surface.
                         */
                        auto const& subGrid = Environment<simDim>::get().SubGrid();
                        auto const globalDomainCells = subGrid.getGlobalDomain().size;
                        auto const direction = getDirection();
                        auto const focus = getFocus();
                        // Value of line parameter p such that origin = line(originP)
                        auto originP = -std::numeric_limits<float_X>::infinity();
                        for(uint32_t axis = 0u; axis < simDim; ++axis)
                        {
                            // Ignore axes with near-absent propagation direction to avoid numerical issues
                            if(std::abs(direction[axis]) > std::numeric_limits<float_X>::epsilon())
                            {
                                // Take into account 0.75 cells inwards shift of Huygens surface
                                auto const minPosition
                                    = (static_cast<float_X>(POSITION[axis][0]) + 0.75_X) * cellSize[axis];
                                auto const maxPositionIdx = (POSITION[axis][1] > 0)
                                    ? POSITION[axis][1]
                                    : globalDomainCells[axis] + POSITION[axis][1];
                                auto const maxPosition
                                    = (static_cast<float_X>(maxPositionIdx) - 0.75_X) * cellSize[axis];
                                /* First we find intersection of line(p) with continuations of the generation planes
                                 * along the axis. Between these two points we choose the smaller parameter value as
                                 * described above. Note that a point line(axisP) does not have to be inside the
                                 * generation surface.
                                 */
                                auto const axisP = std::min(
                                    (minPosition - focus[axis]) / direction[axis],
                                    (maxPosition - focus[axis]) / direction[axis]);
                                /* Here we have to use max of those parameter values as only it will give (after this
                                 * loop is finished) a point at the generation surface. All other axisP values are at
                                 * continuations of the generation places but outside of the surface as a whole.
                                 */
                                originP = std::max(originP, axisP);
                            }
                        }
                        return focus + originP * direction;
                    }

                    /** Get unit axis vectors of internal coordinate system
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

                    /** Get value of phase velocity
                     *
                     * To avoid recalculations, we calculate it once (per given profile parameters) and store
                     * statically.
                     */
                    HINLINE static float_X getPhaseVelocity()
                    {
                        static float_X phaseVelocityValue = detail::calculatePhaseVelocity<Unitless>();
                        return phaseVelocityValue;
                    }

                    //! Check that the input units are valid
                    HINLINE static void checkUnit(float3_64 const unitField)
                    {
                        /* Ensure that we always get unitField = (UNIT_EFIELD, UNIT_EFIELD, UNIT_EFIELD) so that
                         * we can always calculate in internal units and avoid conversions in child types.
                         * We can afford it each time, as this is done on host before kernel.
                         */
                        for(uint32_t axis = 0; axis < 3; axis++)
                        {
                            // In principle 1 ulp should work, but just to be safe against changes in unit system
                            constexpr double ulp = 4.0;
                            constexpr double eps = std::numeric_limits<double>::epsilon();
                            bool const isMatchingUnit
                                = (std::fabs(unitField[axis] - UNIT_EFIELD) <= eps * UNIT_EFIELD * ulp);
                            if(!isMatchingUnit)
                            {
                                throw std::runtime_error(
                                    "Incident field BaseFunctorE created with wrong unit: expected "
                                    + std::to_string(UNIT_EFIELD) + ", got " + std::to_string(unitField[axis]));
                            }
                        }
                    }
                };

                /** Base class for incident E functors of separable lasers
                 *
                 * In internal coordinates these lasers have form
                 * f(time, position) = Longitudinal(time) * Transversal(position)
                 * This class implements a standard workflow for calculating such lasers.
                 *
                 * @tparam T_BaseParam parameter structure matching profiles::BaseParam requirements
                 */
                template<typename T_BaseParam>
                struct BaseSeparableFunctorE : public BaseFunctorE<T_BaseParam>
                {
                    //! Base functor
                    using Base = BaseFunctorE<T_BaseParam>;

                    /** Create a functor on the host side, check that unit matches the internal E unit
                     *
                     * @param currentStep current time step index, note that it is fractional
                     * @param unitField conversion factor from SI to internal units,
                     *                  fieldE_internal = fieldE_SI / unitField
                     */
                    HINLINE BaseSeparableFunctorE(float_X const currentStep, float3_64 const unitField)
                        : Base(currentStep, unitField)
                    {
                    }

                    /** Calculate value of given functor representing a separable laser
                     *
                     * @tparam T_SeparableFunctor functor type, must match interface of Base and define methods
                     *                            getLongitudinal(time, phaseShift), getTransversal(totalCellIdx)
                     *
                     * @param functor functor object
                     * @param totalCellIdx cell index in the total domain
                     */
                    template<typename T_SeparableFunctor>
                    HDINLINE float3_X operator()(T_SeparableFunctor const& functor, floatD_X const& totalCellIdx) const
                    {
                        auto const time = functor.getCurrentTime(totalCellIdx);
                        // Cut off when the laser has not entered at this point yet to avoid confusion.
                        if(time < 0.0_X)
                            return float3_X::create(0.0_X);
                        auto const transversal = functor.getTransversal(totalCellIdx);
                        if(T_SeparableFunctor::Unitless::Polarisation == PolarisationType::Linear)
                            return functor.getLinearPolarizationVector()
                                * (functor.getLongitudinal(time, 0.0_X) * transversal);
                        else
                        {
                            auto const phaseShift = pmacc::math::Pi<float_X>::halfValue;
                            return functor.getCircularPolarizationVector1()
                                * (functor.getLongitudinal(time, phaseShift) * transversal)
                                + functor.getCircularPolarizationVector2()
                                * (functor.getLongitudinal(time, 0.0_X) * transversal);
                        }
                    }
                };

                /** Base class for incident E functors of separable lasers with Gaussian transversal profile
                 *
                 * In internal coordinates these lasers have transversal profile in form
                 * exp(-squared_transversal_distance)
                 *
                 * This base class provides such getTransversal() that also matches requirements of
                 * BaseSeparableFunctorE.
                 *
                 * @tparam T_BaseTransversalGaussianParam parameter structure matching
                 * profiles::T_BaseTransversalGaussianParam requirements
                 */
                template<typename T_BaseTransversalGaussianParam>
                struct BaseSeparableTransversalGaussianFunctorE
                    : public BaseSeparableFunctorE<T_BaseTransversalGaussianParam>
                {
                    //! Base class
                    using Base = BaseSeparableFunctorE<T_BaseTransversalGaussianParam>;

                    //! Unitless parameters type
                    using Unitless
                        = profiles::detail::BaseTransversalGaussianParamUnitless<T_BaseTransversalGaussianParam>;

                    /** Create a functor on the host side, check that unit matches the internal E unit
                     *
                     * @param currentStep current time step index, note that it is fractional
                     * @param unitField conversion factor from SI to internal units,
                     *                  fieldE_internal = fieldE_SI / unitField
                     */
                    HINLINE BaseSeparableTransversalGaussianFunctorE(
                        float_X const currentStep,
                        float3_64 const unitField)
                        : Base(currentStep, unitField)
                    {
                    }

                    /** Calculate value of given functor representing a separable laser with Gaussian transversal
                     * profile
                     *
                     * @tparam T_SeparableFunctor functor type, must match interface of Base and define method
                     *                            getLongitudinal(time, phaseShift)
                     *
                     * @param functor functor object
                     * @param totalCellIdx cell index in the total domain
                     */
                    template<typename T_Functor>
                    HDINLINE float3_X operator()(T_Functor const& functor, floatD_X const& totalCellIdx) const
                    {
                        return Base::operator()(functor, totalCellIdx);
                    }

                    /** Get transversal Gaussian factor for the given position
                     *
                     * Interface required by Base.
                     *
                     * @param totalCellIdx cell index in the total domain (including all moving window slides)
                     */
                    HDINLINE float_X getTransversal(floatD_X const& totalCellIdx) const
                    {
                        float3_X internalPosition = this->getInternalCoordinates(totalCellIdx);
                        internalPosition[0] = 0.0_X;
                        auto const w0 = float3_X(1.0_X, Unitless::W0_AXIS_1, Unitless::W0_AXIS_2);
                        auto const r2 = pmacc::math::abs2(internalPosition / w0);
                        return math::exp(-r2);
                    }
                };

                /** Helper functor to calculate values of B from values of E using slowly varying envelope
                 * approximation (SVEA) for the given axis and direction
                 *
                 * The functor follows FunctorIncidentFieldConcept and thus can be used as FunctorIncidentB.
                 *
                 * @tparam T_FunctorIncidentE functor for the incident E field, follows the interface of
                 *                            FunctorIncidentFieldConcept (defined in Functors.hpp)
                 */
                template<typename T_FunctorIncidentE>
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
                     * k is pulse propagation direction vector
                     * E is value returned by a base functor at the target location and time of resulting B
                     *
                     * @param totalCellIdx cell index in the total domain (including all moving window slides)
                     * @return incident field B value in internal units
                     */
                    HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
                    {
                        // Get corresponding E value, it is already in internal units
                        auto const eValue = Base::operator()(totalCellIdx);
                        return pmacc::math::cross(Base::getDirection(), eValue) / SPEED_OF_LIGHT;
                    }
                };
            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
