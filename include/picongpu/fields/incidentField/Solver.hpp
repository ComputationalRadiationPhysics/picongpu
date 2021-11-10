/* Copyright 2020-2021 Sergei Bastrakov
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

#include "picongpu/fields/Fields.hpp"
#include "picongpu/fields/MaxwellSolver/FDTD/FDTD.def"
#include "picongpu/fields/absorber/Absorber.hpp"
#include "picongpu/fields/incidentField/Profiles.hpp"
#include "picongpu/fields/incidentField/Solver.kernel"
#include "picongpu/fields/incidentField/Traits.hpp"
#include "picongpu/traits/GetCurl.hpp"

#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/traits/IsBaseTemplateOf.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

//! @note this file uses the same naming convention for updated and incident field as Solver.kernel.

namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace detail
            {
                /** Internally used parameters of the incidentField generation normal to the given axis
                 *
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 */
                template<uint32_t T_axis>
                struct Parameters
                {
                    /** Create a parameters instance
                     *
                     * @param cellDescription cell description for kernels
                     */
                    Parameters(MappingDesc const cellDescription) : cellDescription(cellDescription)
                    {
                    }

                    /** Offset of the Huygens surface from min border of the global domain
                     *
                     * The offset is from the start of CORE + BORDER area.
                     * Counted in full cells, the surface is additionally offset by 0.75 cells
                     */
                    pmacc::DataSpace<simDim> offsetMinBorder;

                    /** Offset of the Huygens surface from max border of the global domain
                     *
                     * The offset is from the start of CORE + BORDER area.
                     * Counted in full cells, the surface is additionally offset by 0.75 cells
                     */
                    pmacc::DataSpace<simDim> offsetMaxBorder;

                    /** Direction of the incidentField propagation
                     *
                     * +1._X is positive direction (from the min boundary inwards).
                     * -1._X is negative direction (from the max boundary inwards)
                     */
                    float_X direction;

                    //! Time iteration at which the source incidentField values will be calculated
                    float_X sourceTimeIteration;

                    //! Time increment in the target field, in iterations
                    float_X timeIncrementIteration;

                    //! Whether there is an active (non-none) min source for each axis
                    pmacc::math::Vector<bool, simDim> hasMinSource;

                    //! Cell description for kernels
                    MappingDesc const cellDescription;
                };

                /** Check compile- and run-time requirements for incidentField solver
                 *
                 * Cause compile error or throw when a check is failed.
                 *
                 * @tparam T_UpdateFunctor update functor type
                 *
                 * @param updateFunctor update functor
                 * @param beginLocalUserIdx begin active grid index, in the local domain without guards
                 */
                template<typename T_UpdateFunctor>
                inline void checkRequirements(
                    T_UpdateFunctor const& updateFunctor,
                    pmacc::DataSpace<simDim> const& beginLocalUserIdx)
                {
                    /* Only the FDTD solvers are supported.
                     * Make the condition depend on the template parameters so that it is only compile-time checked
                     * when this part is instantiated, i.e. for non-None sources.
                     */
                    PMACC_CASSERT_MSG(
                        _error_field_solver_does_not_support_incident_field_use_fdtd_solver,
                        pmacc::traits::IsBaseTemplateOf_t<maxwellSolver::FDTD, Solver>::value
                            && (sizeof(T_UpdateFunctor*) != 0));

                    // The implementation assumes the layout of the Yee grid where Ex is at (i + 0.5) cells in x
                    auto const exPosition = traits::FieldPosition<cellType::Yee, FieldE>{}()[0];
                    PMACC_VERIFY_MSG(
                        exPosition[0] == 0.5_X,
                        "incident field source does not support the used Yee grid layout");

                    // Ensure gap along the current axis is large enough that we don't touch the absorber area
                    auto const margin = static_cast<int32_t>(updateFunctor.margin);
                    if((updateFunctor.direction > 0) && (GAP_FROM_ABSORBER[updateFunctor.axis][0] + 1 < margin))
                        throw std::runtime_error(
                            "Incident field GAP_FROM_ABSORBER[" + std::to_string(updateFunctor.axis)
                            + "][0] is too small for used field solver, must be at least "
                            + std::to_string(margin - 1));
                    if((updateFunctor.direction < 0) && (GAP_FROM_ABSORBER[updateFunctor.axis][1] + 1 < margin))
                        throw std::runtime_error(
                            "Incident field GAP_FROM_ABSORBER[" + std::to_string(updateFunctor.axis)
                            + "][1] is too small for used field solver, must be at least "
                            + std::to_string(margin - 1));

                    /* Current implementation requires all updated values (along the active axis) to be inside the same
                     * local domain
                     */
                    auto const& subGrid = Environment<simDim>::get().SubGrid();
                    auto const localDomainSize = subGrid.getLocalDomain().size[updateFunctor.axis];
                    if((beginLocalUserIdx[updateFunctor.axis] + 1 < margin)
                       || (beginLocalUserIdx[updateFunctor.axis] + margin > localDomainSize))
                        throw std::runtime_error(
                            "The Huygens surface for incident field generation is too close to a local domain border."
                            "Adjust GAP_FROM_ABSORBER or grid distribution over gpus.");
                }

                /** Update a field with the given incidentField normally to the given axis
                 *
                 * @note This function operates in terms of the standard Yee field solver.
                 * It concerns calculations, naming, and comments.
                 * The kernel called takes it into account and provides correct processing for all supported solvers.
                 *
                 * @tparam T_UpdatedField updatedField type (FieldE or FieldB)
                 * @tparam T_IncidentField incidentField type (FieldB or FieldE)
                 * @tparam T_CurlIncidentField curl(incidentField) functor type
                 * @tparam T_FunctorIncidentField incidentField source functor type
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 *
                 * @param parameters parameters
                 * @param curlCoefficient coefficient in front of the curl(incidentField) in the Maxwell's equations
                 */
                template<
                    typename T_UpdatedField,
                    typename T_IncidentField,
                    typename T_CurlIncidentField,
                    typename T_FunctorIncidentField,
                    uint32_t T_axis>
                inline void updateField(Parameters<T_axis> const& parameters, float_X const curlCoefficient)
                {
                    /* Whether the field values we are updating are in the total- or scattered-field region:
                     * total field when x component is not on the x cell border,
                     * scattered field when x component is on the x cell border
                     */
                    auto updatedFieldPositions = traits::FieldPosition<cellType::Yee, T_UpdatedField>{}();
                    bool isUpdatedFieldTotal = (updatedFieldPositions[0][0] != 0.0_X);

                    /* Start and end of the source area in the user total coordinates
                     * (the coordinate system in which a user functor is expressed, no guards)
                     */
                    auto const& subGrid = Environment<simDim>::get().SubGrid();
                    auto const globalDomainOffset = subGrid.getGlobalDomain().offset;
                    auto beginUserIdx = parameters.offsetMinBorder + globalDomainOffset;
                    /* Total field in positive direction needs to have the begin adjusted by one to get the first
                     * index inside the respective region.
                     * Here we want to shift only for axes with active positive-direction source.
                     * So for other boundaries the Huygens surface is effectively pushed one cell outside.
                     * We do it to avoid creating artificial non-uniformities at the edges.
                     * (Example: consider there is only XMin source and no YMin, no ZMin.
                     * Then the XMin source is applied for y = beginUserIdx.y() and z = beginUserIdx.z() as usual.
                     * If we always shifted by 1 in all directions, it would have made those layers special
                     * and caused unnecessary edge effects.)
                     */
                    if(isUpdatedFieldTotal && parameters.direction > 0)
                        for(uint32_t d = 0; d < simDim; d++)
                            if(parameters.hasMinSource[d])
                                beginUserIdx[d]++;
                    auto endUserIdx = subGrid.getGlobalDomain().size - parameters.offsetMaxBorder + globalDomainOffset;
                    if(parameters.direction > 0)
                        endUserIdx[T_axis] = beginUserIdx[T_axis] + 1;
                    else
                        beginUserIdx[T_axis] = endUserIdx[T_axis] - 1;

                    // Convert to the local domain indices
                    using Index = pmacc::DataSpace<simDim>;
                    using IntVector = pmacc::math::Vector<int, simDim>;
                    auto const localDomain = subGrid.getLocalDomain();
                    auto const totalCellOffset = globalDomainOffset + localDomain.offset;
                    auto const beginLocalUserIdx
                        = Index{pmacc::math::max(IntVector{beginUserIdx - totalCellOffset}, IntVector::create(0))};
                    auto const endLocalUserIdx = Index{
                        pmacc::math::min(IntVector{endUserIdx - totalCellOffset}, IntVector{localDomain.size})};

                    // Check if we have any active cells in the local domain
                    bool areAnyCellsInLocalDomain = true;
                    for(uint32_t d = 0; d < simDim; d++)
                        areAnyCellsInLocalDomain
                            = areAnyCellsInLocalDomain && (beginLocalUserIdx[d] < endLocalUserIdx[d]);
                    if(!areAnyCellsInLocalDomain)
                        return;

                    // The block-thread organization is same as used for the laser kernel
                    using PlaneSizeInSuperCells = typename pmacc::math::CT::AssignIfInRange<
                        typename SuperCellSize::vector_type,
                        bmpl::integral_c<uint32_t, T_axis>,
                        bmpl::integral_c<int, 1>>::type;
                    auto const superCellSize = SuperCellSize::toRT();
                    auto const gridBlocks
                        = (endLocalUserIdx - beginLocalUserIdx + superCellSize - Index::create(1)) / superCellSize;
                    constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                        pmacc::math::CT::volume<PlaneSizeInSuperCells>::type::value>::value;

                    // Shift by guard size to go to the in-kernel coordinate system
                    auto const mapper = pmacc::makeAreaMapper<CORE + BORDER>(parameters.cellDescription);
                    auto numGuardCells = mapper.getGuardingSuperCells() * SuperCellSize::toRT();
                    auto beginGridIdx = beginLocalUserIdx + numGuardCells;
                    auto endGridIdx = endLocalUserIdx + numGuardCells;

                    // Indexing is done, now prepare the update functor
                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto& updatedField = *dc.get<T_UpdatedField>(T_UpdatedField::getName(), true);
                    auto dataBox = updatedField.getDeviceDataBox();
                    auto const& incidentField = *dc.get<T_IncidentField>(T_IncidentField::getName(), true);
                    using Functor
                        = UpdateFunctor<decltype(dataBox), T_CurlIncidentField, T_FunctorIncidentField, T_axis>;
                    auto functor = Functor{incidentField.getUnit()};
                    functor.updatedField = dataBox;
                    functor.currentStep = parameters.sourceTimeIteration;
                    functor.isUpdatedFieldTotal = isUpdatedFieldTotal;
                    functor.direction = parameters.direction;
                    /* Shift between local grid idx and fractional total cell idx that a user functor needs:
                     * total cell idx = local grid idx + functor.gridIdxShift.
                     */
                    functor.gridIdxShift = totalCellOffset - numGuardCells;

                    /* Compute which components of the incidentField are used,
                     * which components of the updatedField they contribute to and with which coefficients.
                     */

                    // dir0 is boundary normal axis, dir1 and dir2 are two other axes
                    constexpr auto dir0 = T_axis;
                    constexpr auto dir1 = (dir0 + 1) % 3;
                    constexpr auto dir2 = (dir0 + 2) % 3;

                    /* IncidentField components to be used for the two terms.
                     * Note the intentional cross combination here, the following calculations rely on it
                     */
                    functor.incidentComponent1 = dir2;
                    functor.incidentComponent2 = dir1;

                    // Coefficients for the respective terms
                    float_X const coeffBase = curlCoefficient / cellSize[T_axis];
                    functor.coeff1[dir1] = coeffBase;
                    functor.coeff2[dir2] = -coeffBase;

                    /* For the positive direction, the updated total field index was shifted by 1 earlier.
                     * This index shift is translated to in-cell shift for the incidentField here.
                     */
                    auto incidentFieldBaseShift = floatD_X::create(0.0_X);
                    if(parameters.direction > 0)
                    {
                        if(isUpdatedFieldTotal)
                            incidentFieldBaseShift[dir0] = -1.0_X;
                        else
                            incidentFieldBaseShift[dir0] = 1.0_X;
                    }
                    auto incidentFieldPositions = traits::FieldPosition<cellType::Yee, T_IncidentField>{}();
                    functor.inCellShift1 = incidentFieldBaseShift + incidentFieldPositions[functor.incidentComponent1];
                    functor.inCellShift2 = incidentFieldBaseShift + incidentFieldPositions[functor.incidentComponent2];

                    // Check that incidentField can be applied
                    checkRequirements(functor, beginLocalUserIdx);

                    PMACC_KERNEL(ApplyIncidentFieldKernel<numWorkers, PlaneSizeInSuperCells>{})
                    (gridBlocks, numWorkers)(functor, beginGridIdx, endGridIdx);
                }

                /** Update a field with the given incidentField normally to the given axis
                 *
                 * After the sliding window started moving, does nothing for the y boundaries.
                 *
                 * @tparam T_UpdatedField updatedField type (FieldE or FieldB)
                 * @tparam T_IncidentField incidentField type (FieldB or FieldE)
                 * @tparam T_Curl curl(incidentField) functor type
                 * @tparam T_FunctorIncidentField incidentField source functor type
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 *
                 * @param parameters parameters
                 * @param curlCoefficient coefficient in front of the curl(incidentField) in the Maxwell's equations
                 */
                template<
                    typename T_UpdatedField,
                    typename T_IncidentField,
                    typename T_Curl,
                    typename T_FunctorIncidentField,
                    uint32_t T_axis>
                inline void callUpdateField(Parameters<T_axis> const& parameters, float_X const curlCoefficient)
                {
                    // IncidentField generation at y boundaries cannot be performed once the window started moving
                    const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(
                        static_cast<uint32_t>(parameters.sourceTimeIteration));
                    bool const boxHasSlided = (numSlides != 0);
                    if(!((T_axis == 1) && boxHasSlided))
                        updateField<T_UpdatedField, T_IncidentField, T_Curl, T_FunctorIncidentField>(
                            parameters,
                            curlCoefficient);
                }

                /** Functor to apply contribution of the incidentField functor source to the E field
                 *
                 * @tparam T_Source source type: Source<...> or None
                 */
                template<typename T_Source>
                struct UpdateE;

                //! Functor to apply contribution of the None incidentField functor to the E field
                template<>
                struct UpdateE<None>
                {
                    /** Apply contribution of the None incidentField functor to the E field
                     *
                     * @tparam T_Parameters parameters type
                     */
                    template<typename T_Parameters>
                    void operator()(T_Parameters const&)
                    {
                    }
                };

                /** Functor to apply contribution of the incidentField source functor to the E field
                 *
                 * @tparam T_FunctorIncidentE functor for the incident E field (not used)
                 * @tparam T_FunctorIncidentB functor for the incident B field
                 */
                template<typename T_FunctorIncidentE, typename T_FunctorIncidentB>
                struct UpdateE<Source<T_FunctorIncidentE, T_FunctorIncidentB>>
                {
                    /** Apply contribution of the incidentField source functor to the E field
                     *
                     * @tparam T_Parameters parameters type
                     *
                     * @param parameters parameters
                     */
                    template<typename T_Parameters>
                    void operator()(T_Parameters const& parameters)
                    {
                        auto const timeIncrement = parameters.timeIncrementIteration * DELTA_T;
                        /* The update is structurally
                         * E(t + timeIncrement) = E(t) + timeIncrement * c2 * curl(B(t + timeIncrement/2))
                         */
                        constexpr auto c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
                        auto const curlCoefficient = timeIncrement * c2;
                        using UpdatedField = picongpu::FieldE;
                        using IncidentField = picongpu::FieldB;
                        using Curl = traits::GetCurlB<Solver>::type;
                        callUpdateField<UpdatedField, IncidentField, Curl, T_FunctorIncidentB>(
                            parameters,
                            curlCoefficient);
                    }
                };

                /** Functor to apply contribution of the incidentField source functor to the B field
                 *
                 * @tparam T_Source source type: Source<...> or None
                 */
                template<typename T_Source>
                struct UpdateB;

                //! Functor to apply contribution of the None incidentField functor to the B field
                template<>
                struct UpdateB<None>
                {
                    /** Apply contribution of the None incidentField functor to the B field
                     *
                     * @tparam T_Parameters parameters type
                     */
                    template<typename T_Parameters>
                    void operator()(T_Parameters const&)
                    {
                    }
                };

                /** Functor to apply contribution of the incidentField source functor to the B field
                 *
                 * @tparam T_FunctorIncidentE functor for the incident E field
                 * @tparam T_FunctorIncidentB functor for the incident B field (not used)
                 */
                template<typename T_FunctorIncidentE, typename T_FunctorIncidentB>
                struct UpdateB<Source<T_FunctorIncidentE, T_FunctorIncidentB>>
                {
                    /** Apply contribution of the incidentField source functor to the B field
                     *
                     * @tparam T_Parameters parameters type
                     *
                     * @param parameters parameters
                     */
                    template<typename T_Parameters>
                    void operator()(T_Parameters const& parameters)
                    {
                        auto const timeIncrement = parameters.timeIncrementIteration * DELTA_T;
                        /* The update is structurally
                         * B(t + timeIncrement) = B(t) - timeIncrement * curl(E(t + timeIncrement/2))
                         */
                        auto const curlCoefficient = -timeIncrement;
                        using UpdatedField = picongpu::FieldB;
                        using IncidentField = picongpu::FieldE;
                        using Curl = traits::GetCurlE<Solver>::type;
                        callUpdateField<UpdatedField, IncidentField, Curl, T_FunctorIncidentE>(
                            parameters,
                            curlCoefficient);
                    }
                };

            } // namespace detail

            /** Solver for incident fields to be called inside an FDTD Maxwell's solver
             *
             * It uses the total field / scattered field technique for FDTD solvers.
             * Implementation is based on
             * M. Potter, J.-P. Berenger. A Review of the Total Field/Scattered Field Technique for the FDTD Method.
             * FERMAT, Volume 19, Article 1, 2017.
             *
             * The simulation area is virtually divided into two regions with a so-called Huygens surface.
             * It is composed of six axis-aligned plane segments in 3d or four axis-aligned line segments in 2d.
             * So it is parallel to the interface between the absorber area and internal non-absorbing area.
             * The position of the Huygens surface is controlled by offset from the interface inwards.
             * This offset is a sum of the user-specified gap and 0.75 of a cell into the internal area from each side.
             * (The choice of 0.75 is arbitrary by this implementation, only required to be not in full or half cells).
             * All gaps >= 0 are supported if the surface covers an internal volume of at least one full cell.
             *
             * In the internal volume bounded by the Huygens surface, the E and B fields are so-called full fields.
             * They are a combined result of incoming incident fields and the internally happening dynamics.
             * So these are normal fields used in a PIC simulation.
             * Outside of the surface, the fields do not have the incident part, they are so-called scattered fields.
             * Thus, there should normally be no relevant field-related physics happening there.
             * Practically, for interpretation of the output a user could view this area as if the absorber was thicker
             * in a way that the fields there are not immediately related to the fields in the internal area.
             */
            class Solver
            {
            public:
                /** Create a solver instance
                 *
                 * @param cellDescription cell description for kernels
                 */
                Solver(MappingDesc const cellDescription) : cellDescription(cellDescription)
                {
                    /* Compute offsets from global domain borders, without guards.
                     * These can end up being outside of the local domain, it is handled later.
                     */
                    auto const& absorber = fields::absorber::Absorber::get();
                    absorber::Thickness absorberThickness = absorber.getGlobalThickness();
                    for(uint32_t axis = 0u; axis < simDim; ++axis)
                    {
                        offsetMinBorder[axis] = absorberThickness(axis, 0) + GAP_FROM_ABSORBER[axis][0];
                        offsetMaxBorder[axis] = absorberThickness(axis, 1) + GAP_FROM_ABSORBER[axis][1];
                    }
                    hasMinSource[0] = !std::is_same<XMin, None>::value;
                    hasMinSource[1] = !std::is_same<YMin, None>::value;
                    if(simDim == 3)
                        hasMinSource[2] = !std::is_same<ZMin, None>::value;
                }

                /** Apply contribution of the incident B field to the E field update by one time step
                 *
                 * Must be called together with the updateE in the field solver.
                 *
                 * @param sourceTimeIteration time iteration at which the source incident B field
                 *                            (not the target E field!) values will be calculated
                 */
                void updateE(float_X const sourceTimeIteration)
                {
                    updateE<0, XMin, XMax>(sourceTimeIteration);
                    updateE<1, YMin, YMax>(sourceTimeIteration);
                    updateE<2, ZMin, ZMax>(sourceTimeIteration);
                }

                /** Apply contribution of the incident E field to the B field update by half a time step
                 *
                 * Must be called together with the updateBHalf in the field solver.
                 *
                 * @param sourceTimeIteration time iteration at which the source incident E field
                 *                            (not the target B field!) values will be calculated
                 * @warning when calling this method for the second half of the B field update, the sourceTimeIteration
                 *          value must be the same as for the first half. Otherwise the halves would not match each
                 *          other, and it will cause errors by a similar mechanism as #3418 did for PML.
                 */
                void updateBHalf(float_X const sourceTimeIteration)
                {
                    updateBHalf<0, XMin, XMax>(sourceTimeIteration);
                    updateBHalf<1, YMin, YMax>(sourceTimeIteration);
                    updateBHalf<2, ZMin, ZMax>(sourceTimeIteration);
                }

            private:
                /** Apply contribution of the incident B field to the E field update by one time step
                 *
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_MinSource source type for the min boundary along the axis: Source<...> or None
                 * @tparam T_MaxSource source type for the max boundary along the axis: Source<...> or None
                 *
                 * @param sourceTimeIteration time iteration at which the source incident B field
                 *                            (not the target E field!) values will be calculated
                 */
                template<uint32_t T_axis, typename T_MinSource, typename T_MaxSource>
                void updateE(float_X const sourceTimeIteration)
                {
                    auto parameters = detail::Parameters<T_axis>{cellDescription};
                    parameters.offsetMinBorder = offsetMinBorder;
                    parameters.offsetMaxBorder = offsetMaxBorder;
                    parameters.hasMinSource = hasMinSource;
                    parameters.direction = 1.0_X;
                    parameters.sourceTimeIteration = sourceTimeIteration;
                    parameters.timeIncrementIteration = 1.0_X;
                    using UpdateMin = typename detail::UpdateE<T_MinSource>;
                    UpdateMin{}(parameters);
                    parameters.direction = -1.0_X;
                    using UpdateMax = typename detail::UpdateE<T_MaxSource>;
                    UpdateMax{}(parameters);
                }

                /** Apply contribution of the incident E field to the B field update by half a time step
                 *
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_MinSource source type for the min boundary along the axis: Source<...> or None
                 * @tparam T_MaxSource source type for the max boundary along the axis: Source<...> or None
                 *
                 * @param sourceTimeIteration time iteration at which the source incident E field
                 *                            (not the target B field!) values will be calculated
                 */
                template<uint32_t T_axis, typename T_MinSource, typename T_MaxSource>
                void updateBHalf(float_X const sourceTimeIteration)
                {
                    auto parameters = detail::Parameters<T_axis>{cellDescription};
                    parameters.offsetMinBorder = offsetMinBorder;
                    parameters.offsetMaxBorder = offsetMaxBorder;
                    parameters.hasMinSource = hasMinSource;
                    parameters.direction = 1.0_X;
                    parameters.sourceTimeIteration = sourceTimeIteration;
                    parameters.timeIncrementIteration = 0.5_X;
                    using UpdateMin = typename detail::UpdateB<T_MinSource>;
                    UpdateMin{}(parameters);
                    parameters.direction = -1.0_X;
                    using UpdateMax = typename detail::UpdateB<T_MaxSource>;
                    UpdateMax{}(parameters);
                }

                //! ZMin type to be used, shadows ZMin from .param on purpose to handle 2d and 3d uniformly
                using ZMin = detail::ZMin;

                //! ZMax type to be used, shadows ZMax from .param on purpose to handle 2d and 3d uniformly
                using ZMax = detail::ZMax;

                /** Offset of the Huygens surface from min border of the global domain
                 *
                 * The offset is from the start of CORE + BORDER area.
                 * Counted in full cells, the surface is additionally offset by 0.75 cells
                 */
                pmacc::DataSpace<simDim> offsetMinBorder;

                /** Offset of the Huygens surface from max border of the global domain
                 *
                 * The offset is from the end of CORE + BORDER area.
                 * Counted in full cells, the surface is additionally offset by 0.75 cells
                 */
                pmacc::DataSpace<simDim> offsetMaxBorder;

                //! Whether there is an active (non-none) min source for each axis
                pmacc::math::Vector<bool, simDim> hasMinSource;

                //! Cell description for kernels
                MappingDesc const cellDescription;
            };

        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
