/* Copyright 2020-2023 Sergei Bastrakov
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
#include "picongpu/fields/MaxwellSolver/GetTimeStep.hpp"
#include "picongpu/fields/MaxwellSolver/Substepping/Substepping.def"
#include "picongpu/fields/absorber/Absorber.hpp"
#include "picongpu/fields/incidentField/Functors.hpp"
#include "picongpu/fields/incidentField/Solver.kernel"
#include "picongpu/fields/incidentField/Traits.hpp"
#include "picongpu/fields/incidentField/profiles/profiles.hpp"
#include "picongpu/traits/GetCurl.hpp"

#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/traits/IsBaseTemplateOf.hpp>

#include <algorithm>
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

                    /** Total position of the Huygens surface for min and max borders
                     *
                     * It is offset from the origin of the user coordinate system, in cells.
                     * The Huygens surface is additionally offset by 0.75 cells inwards from this position.
                     *
                     * @{
                     */
                    pmacc::DataSpace<simDim> totalPositionMinBorder;
                    pmacc::DataSpace<simDim> totalPositionMaxBorder;
                    /** @} */

                    /** Direction of the incidentField propagation
                     *
                     * +1._X is positive direction (from the min boundary inwards).
                     * -1._X is negative direction (from the max boundary inwards)
                     */
                    float_X direction;

                    //! Time iteration at which the source incidentField values will be calculated
                    float_X sourceTimeIteration;

                    //! Time increment in the target field (in time, not in iterations)
                    float_X timeIncrement;

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
                        "incident field profile does not support the used Yee grid layout");

                    // Ensure offset along the current axis is large enough that we don't touch the absorber area
                    auto const axis = updateFunctor.axis;
                    auto const margin = static_cast<int32_t>(updateFunctor.margin);
                    // Index of the boundary in 2d arrays like absorber thickness
                    auto const boundaryIdx = (updateFunctor.direction > 0) ? 0 : 1;
                    auto const& absorber = fields::absorber::Absorber::get();
                    auto const absorberThickness = absorber.getGlobalThickness()(axis, boundaryIdx);
                    auto const minAllowedOffsetFromBoundary = static_cast<int32_t>(absorberThickness + margin - 1);

                    // Calculate offset of Huygens surface from the active boundary in the inwards direction
                    auto const& subGrid = Environment<simDim>::get().SubGrid();
                    int32_t offset = 0;
                    if(boundaryIdx == 0)
                        offset = POSITION[axis][boundaryIdx];
                    else if(POSITION[axis][boundaryIdx] > 0)
                        offset = subGrid.getGlobalDomain().size[axis] - POSITION[axis][boundaryIdx];
                    else
                        offset = -POSITION[axis][boundaryIdx];

                    // For YMax boundary and moving window on we allow positioning outside of global volume
                    auto const movingWindowEnabled = MovingWindow::getInstance().isEnabled();
                    bool const skipOffsetCheck = ((axis == 1) && (boundaryIdx == 1) && movingWindowEnabled);
                    if(skipOffsetCheck)
                        offset = minAllowedOffsetFromBoundary;

                    if(offset < minAllowedOffsetFromBoundary)
                        throw std::runtime_error(
                            "Incident field POSITION[" + std::to_string(axis) + "][" + std::to_string(boundaryIdx)
                            + "] is too close to the boundary for used field solver and absorber, must be at least "
                            + std::to_string(minAllowedOffsetFromBoundary) + " cells away");

                    /* Current implementation requires all updated values (along the active axis) to be inside the same
                     * local domain
                     */
                    auto const localDomainSize = subGrid.getLocalDomain().size[axis];
                    if((beginLocalUserIdx[axis] + 1 < margin) || (beginLocalUserIdx[axis] + margin > localDomainSize))
                        throw std::runtime_error(
                            "The Huygens surface for incident field generation is too close to a local domain border."
                            "Adjust POSITION or grid distribution over gpus.");
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
                 * @param transversalContiguousHuygensSurface true if the is allowed to be stretched to support
                 *                                            periodic boundary condition, else false
                 */
                template<
                    typename T_UpdatedField,
                    typename T_IncidentField,
                    typename T_CurlIncidentField,
                    typename T_FunctorIncidentField,
                    uint32_t T_axis>
                inline void updateField(
                    Parameters<T_axis> const& parameters,
                    float_X const curlCoefficient,
                    bool transversalContiguousHuygensSurface)
                {
                    /* Whether the field values we are updating are in the total- or scattered-field region:
                     * total field when x component is not on the x cell border,
                     * scattered field when x component is on the x cell border
                     */
                    auto updatedFieldPositions = traits::FieldPosition<cellType::Yee, T_UpdatedField>{}();
                    bool isUpdatedFieldTotal = (updatedFieldPositions[0][0] != 0.0_X);

                    /* Start and end of the source area in the user total coordinates.
                     * Add extra 1 to account for 0.75 Huygens surface shift.
                     * However, the shift is reverted for min-border row of B due to our Yee grid configuration.
                     */
                    auto beginUserIdx = parameters.totalPositionMinBorder + pmacc::DataSpace<simDim>::create(1);
                    if(!isUpdatedFieldTotal && (parameters.direction > 0))
                        beginUserIdx[T_axis] -= 1;
                    auto endUserIdx = parameters.totalPositionMaxBorder;

                    // Prepare update functor type
                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto const& incidentField = *dc.get<T_IncidentField>(T_IncidentField::getName());
                    using Functor = UpdateFunctor<T_CurlIncidentField, T_FunctorIncidentField, T_axis>;
                    auto functor = Functor{
                        parameters.sourceTimeIteration,
                        parameters.direction,
                        curlCoefficient,
                        incidentField.getUnit()};

                    functor.isUpdatedFieldTotal = isUpdatedFieldTotal;

                    // Convert to the local domain indices
                    auto const& subGrid = Environment<simDim>::get().SubGrid();

                    // Set local domain information
                    auto& gridController = pmacc::Environment<simDim>::get().GridController();
                    auto const localDomainIdx = gridController.getPosition();
                    auto const numLocalDomains = gridController.getGpuNodes();

                    const DataSpace<DIM3> periodic
                        = Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();

                    for(uint32_t d = 0u; d < simDim; ++d)
                    {
                        bool isLaserAxisDirection = d == T_axis;

                        if(transversalContiguousHuygensSurface && periodic[d] == 1 && !isLaserAxisDirection)
                        {
                            /* Extend the Huygens surface begin and end boundaries to all cells in the periodic
                             * direction. If the laser direction is periodic do not move Huyhens surface to zero and
                             * outer simulation domain else the user laser definition will silently adjusted.
                             */
                            beginUserIdx[d] = 0;
                            endUserIdx[d] = subGrid.getGlobalDomain().offset[d] + subGrid.getGlobalDomain().size[d];
                            functor.isLastLocalDomain[d] = false;
                        }
                        else
                        {
                            /* for non periodic directions we need to handle special cases for the last cell of the
                             * direction on the outer surface in the kernel.
                             * see: https://github.com/ComputationalRadiationPhysics/picongpu/pull/4298
                             */
                            functor.isLastLocalDomain[d] = (localDomainIdx[d] == numLocalDomains[d] - 1);
                        }
                    }

                    constexpr int margin = Functor::margin;
                    constexpr int sizeAlongAxis = 2 * margin - 1;

                    if(parameters.direction > 0)
                    {
                        beginUserIdx[T_axis] -= (margin - 1);
                        endUserIdx[T_axis] = beginUserIdx[T_axis] + sizeAlongAxis;
                    }
                    else
                    {
                        endUserIdx[T_axis] += (margin - 1);
                        beginUserIdx[T_axis] = endUserIdx[T_axis] - sizeAlongAxis;
                    }

                    // Convert to the local domain indices
                    auto const globalDomainOffset = subGrid.getGlobalDomain().offset;
                    auto const localDomain = subGrid.getLocalDomain();
                    auto const totalCellOffset = globalDomainOffset + localDomain.offset;
                    using Index = pmacc::DataSpace<simDim>;
                    using IntVector = pmacc::math::Vector<int, simDim>;
                    auto const beginLocalUserIdx
                        = Index{math::max(IntVector{beginUserIdx - totalCellOffset}, IntVector::create(0))};
                    auto const endLocalUserIdx
                        = Index{math::min(IntVector{endUserIdx - totalCellOffset}, IntVector{localDomain.size})};

                    // Check if we have any active cells in the local domain
                    bool areAnyCellsInLocalDomain = true;
                    for(uint32_t d = 0; d < simDim; d++)
                        areAnyCellsInLocalDomain
                            = areAnyCellsInLocalDomain && (beginLocalUserIdx[d] < endLocalUserIdx[d]);
                    if(!areAnyCellsInLocalDomain)
                        return;

                    /* The block size is generally equal to supercell size, but can be smaller along T_axis
                     * when there is not enough work
                     */
                    constexpr int supercellSizeAlongAxis
                        = pmacc::math::CT::At_c<SuperCellSize::vector_type, T_axis>::type::value;
                    constexpr int blockSizeAlongAxis = std::min(sizeAlongAxis, supercellSizeAlongAxis);
                    using BlockSize = typename pmacc::math::CT::AssignIfInRange<
                        typename SuperCellSize::vector_type,
                        std::integral_constant<uint32_t, T_axis>,
                        std::integral_constant<int, blockSizeAlongAxis>>::type;
                    auto const superCellSize = SuperCellSize::toRT();
                    auto const gridBlocks
                        = (endLocalUserIdx - beginLocalUserIdx + superCellSize - Index::create(1)) / superCellSize;

                    // Shift by guard size to go to the in-kernel coordinate system
                    auto const mapper = pmacc::makeAreaMapper<CORE + BORDER>(parameters.cellDescription);
                    auto numGuardCells = mapper.getGuardingSuperCells() * SuperCellSize::toRT();
                    auto beginGridIdx = beginLocalUserIdx + numGuardCells;
                    auto endGridIdx = endLocalUserIdx + numGuardCells;

                    // Indexing is done, now go on with preparing the update functor

                    /* Shift between local grid idx and fractional total cell idx that a user functor needs:
                     * total cell idx = local grid idx + functor.gridIdxShift.
                     */
                    functor.gridIdxShift = totalCellOffset - numGuardCells;

                    /* For the positive direction, the updated total field index was shifted by 1 earlier.
                     * This index shift is translated to in-cell shift for the incidentField here.
                     */
                    auto incidentFieldBaseShift = floatD_X::create(0.0_X);
                    if(parameters.direction > 0)
                    {
                        if(isUpdatedFieldTotal)
                            incidentFieldBaseShift[T_axis] = -1.0_X;
                        else
                            incidentFieldBaseShift[T_axis] = 1.0_X;
                    }
                    auto incidentFieldPositions = traits::FieldPosition<cellType::Yee, T_IncidentField>{}();
                    functor.inCellShift1 = incidentFieldBaseShift + incidentFieldPositions[functor.incidentComponent1];
                    functor.inCellShift2 = incidentFieldBaseShift + incidentFieldPositions[functor.incidentComponent2];

                    // Check that incidentField can be applied
                    checkRequirements(functor, beginLocalUserIdx);

                    auto& updatedField = *dc.get<T_UpdatedField>(T_UpdatedField::getName());

                    auto workerCfg = lockstep::makeWorkerCfg(BlockSize{});
                    PMACC_LOCKSTEP_KERNEL(ApplyIncidentFieldKernel<BlockSize>{}, workerCfg)
                    (gridBlocks)(functor, updatedField.getDeviceDataBox(), beginGridIdx, endGridIdx);
                }

                /** Functor to update a field with the given incidentField normally to the given axis
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
                    typename T_FunctorIncidentField>
                struct CallUpdateField
                {
                    template<uint32_t T_axis>
                    void operator()(
                        Parameters<T_axis> const& parameters,
                        float_X const curlCoefficient,
                        bool transversalContiguousHuygensSurface)
                    {
                        updateField<T_UpdatedField, T_IncidentField, T_Curl, T_FunctorIncidentField>(
                            parameters,
                            curlCoefficient,
                            transversalContiguousHuygensSurface);
                    }
                };

                /** Partial specialization of functor to update a field with the given incidentField normally to the
                 * given axis for zero functor
                 *
                 * The general implementation works in this case and does nothing as it should.
                 * We provide specialization to avoid compiling and running in this case and short-circuit to doing
                 * nothing.
                 *
                 * @tparam T_UpdatedField updatedField type (FieldE or FieldB)
                 * @tparam T_IncidentField incidentField type (FieldB or FieldE)
                 * @tparam T_Curl curl(incidentField) functor type
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 */
                template<typename T_UpdatedField, typename T_IncidentField, typename T_Curl>
                struct CallUpdateField<T_UpdatedField, T_IncidentField, T_Curl, ZeroFunctor>
                {
                    template<uint32_t T_axis>
                    void operator()(
                        Parameters<T_axis> const& /* parameters */,
                        float_X const /* curlCoefficient */,
                        bool /*transversalContiguousHuygensSurface*/)
                    {
                    }
                };

                /** Functor to apply contribution of the incidentField functor source to the E field
                 *
                 * @tparam T_FunctorIncidentB incident B source functor type
                 */
                template<typename T_FunctorIncidentB>
                struct UpdateE
                {
                    /** Apply contribution of the incidentField source functor to the E field
                     *
                     * @tparam T_Parameters parameters type
                     *
                     * @param parameters parameters
                     * @param transversalContiguousHuygensSurface true if the is allowed to be
                     *                                            stretched to support periodic boundary condition,
                     *                                            else false
                     */
                    template<typename T_Parameters>
                    void operator()(T_Parameters const& parameters, bool transversalContiguousHuygensSurface)
                    {
                        /* The update is structurally
                         * E(t + timeIncrement) = E(t) + timeIncrement * c2 * curl(B(t + timeIncrement/2))
                         */
                        constexpr auto c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
                        auto const curlCoefficient = parameters.timeIncrement * c2;
                        using UpdatedField = picongpu::FieldE;
                        using IncidentField = picongpu::FieldB;
                        using Curl = traits::GetCurlB<Solver>::type;
                        CallUpdateField<UpdatedField, IncidentField, Curl, T_FunctorIncidentB>{}(
                            parameters,
                            curlCoefficient,
                            transversalContiguousHuygensSurface);
                    }
                };

                /** Functor to apply contribution of the incidentField source functor to the B field
                 *
                 * @tparam T_FunctorIncidentE incident E source functor type
                 */
                template<typename T_FunctorIncidentE>
                struct UpdateB
                {
                    /** Apply contribution of the incidentField source functor to the B field
                     *
                     * @tparam T_Parameters parameters type
                     *
                     * @param parameters parameters
                     * @param transversalContiguousHuygensSurface true if the is allowed to be stretched to support
                     *                                            periodic boundary condition, else false
                     */
                    template<typename T_Parameters>
                    void operator()(T_Parameters const& parameters, bool transversalContiguousHuygensSurface)
                    {
                        /* The update is structurally
                         * B(t + timeIncrement) = B(t) - timeIncrement * curl(E(t + timeIncrement/2))
                         */
                        auto const curlCoefficient = -parameters.timeIncrement;
                        using UpdatedField = picongpu::FieldB;
                        using IncidentField = picongpu::FieldE;
                        using Curl = traits::GetCurlE<Solver>::type;
                        CallUpdateField<UpdatedField, IncidentField, Curl, T_FunctorIncidentE>{}(
                            parameters,
                            curlCoefficient,
                            transversalContiguousHuygensSurface);
                    }
                };

            } // namespace detail

            /** Solver for incident fields to be called inside an FDTD Maxwell's solver
             *
             * It uses the total field / scattered field technique for FDTD solvers.
             * Implementation is based on two sources:
             * A. Taflove, S.C. Hagness. Computational Electrodynamics. The Finite-Difference Time-Domain Method. Third
             * Edition. Artech house (2005). Chapter 5.
             * M. Potter, J.-P. Berenger. A Review of the Total Field/Scattered Field Technique for the FDTD Method.
             * FERMAT, Volume 19, Article 1, 2017.
             *
             * The simulation area is virtually divided into two regions with a so-called Huygens surface.
             * It is composed of six axis-aligned plane segments in 3d or four axis-aligned line segments in 2d.
             * So it is parallel to the interface between the absorber area and internal non-absorbing area.
             * The position of the Huygens surface is controlled by offset from the interface inwards.
             * Extra 0.75 of a cell shift is applied towards the internal area from each side.
             * (The choice of 0.75 is arbitrary by this implementation, only required to be not in full or half cells).
             * Thus, the positioning and indexing matches that of [Taflove, Hagness].
             * Offsets must be such that the Huygens surface is located outside of field absorber area and covers an
             * internal volume of at least one full cell.
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
                    auto const& subGrid = Environment<simDim>::get().SubGrid();
                    auto const globalDomainSize = subGrid.getGlobalDomain().size;
                    for(uint32_t axis = 0u; axis < simDim; ++axis)
                    {
                        totalPositionMinBorder[axis] = POSITION[axis][0];
                        // Treat negative right-side positions as described in the .param file
                        if(POSITION[axis][1] > 0)
                            totalPositionMaxBorder[axis] = POSITION[axis][1];
                        else
                            totalPositionMaxBorder[axis] = globalDomainSize[axis] + POSITION[axis][1];
                    }
                    checkPositioning();
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
                    updateE<0, XMinProfiles, XMaxProfiles>(sourceTimeIteration);
                    updateE<1, YMinProfiles, YMaxProfiles>(sourceTimeIteration);
                    updateE<2, ZMinProfiles, ZMaxProfiles>(sourceTimeIteration);
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
                    updateBHalf<0, XMinProfiles, XMaxProfiles>(sourceTimeIteration);
                    updateBHalf<1, YMinProfiles, YMaxProfiles>(sourceTimeIteration);
                    updateBHalf<2, ZMinProfiles, ZMaxProfiles>(sourceTimeIteration);
                }

            private:
                /** Check if Huygens surface positioning is reasonable, print a warning otherwise
                 *
                 * Check that the position is inside the simulation volume (taking into account potential moving
                 * window). Also check that the internal volume is not zero.
                 *
                 * Note that it only checks some conditions for positioning.
                 * So this check is necessary, but not sufficient to ensure a valid configuration.
                 * An incident field solver makes a different check later and throws if it fails.
                 *
                 * This check is skipped when all profiles are None.
                 */
                void checkPositioning() const
                {
                    // Skip the check when no incident field sources enabled
                    if(!isEnabled())
                        return;

                    // Do the check once as its result is always same
                    static bool checkDone = false;
                    if(checkDone)
                        return;
                    checkDone = true;
                    bool isPrinting = (Environment<simDim>::get().GridController().getGlobalRank() == 0);
                    if(isPrinting)
                    {
                        auto const movingWindowEnabled = MovingWindow::getInstance().isEnabled();
                        auto const& subGrid = Environment<simDim>::get().SubGrid();
                        auto const globalDomainSize = subGrid.getGlobalDomain().size;
                        for(uint32_t axis = 0; axis < simDim; axis++)
                        {
                            if(totalPositionMinBorder[axis] < 0)
                                log<picLog::PHYSICS>(
                                    "Warning: Huygens surface at Min border is located outside of simulation volume, "
                                    "no incident field will be generated at the external part of the surface\n");
                            // For moving window we allow for YMax positioning outside of the domain
                            bool const skipMaxCheck = ((axis == 1) && movingWindowEnabled);
                            if(!skipMaxCheck && (totalPositionMaxBorder[axis] >= globalDomainSize[axis]))
                                log<picLog::PHYSICS>(
                                    "Warning: Huygens surface at Max border is located outside of simulation volume, "
                                    "no incident field will be generated at the external part of the surface\n");
                            if(totalPositionMaxBorder[axis] - totalPositionMinBorder[axis] < 2)
                            {
                                log<picLog::PHYSICS>(
                                    "Warning: volume bounded by the Huygens surface is zero, no incident "
                                    "field will be generated\n");
                                break;
                            }
                        }
                    }
                }

                //! Return if incident field is enabled in the simulation i.e. there exists a non-None profile
                static bool isEnabled()
                {
                    using Disabled = pmacc::MakeSeq_t<profiles::None>;
                    auto const isEnabledX
                        = !(std::is_same_v<XMinProfiles, Disabled> && std::is_same_v<XMaxProfiles, Disabled>);
                    auto const isEnabledY
                        = !(std::is_same_v<YMinProfiles, Disabled> && std::is_same_v<YMaxProfiles, Disabled>);
                    auto const isEnabledZ
                        = !(std::is_same_v<ZMinProfiles, Disabled> && std::is_same_v<ZMaxProfiles, Disabled>);
                    return isEnabledX || isEnabledY || isEnabledZ;
                }

                /** Apply contribution of the incident B field to the E field update by one time step
                 *
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_MinProfiles typelist of profiles for the min boundary along the axis
                 * @tparam T_MaxProfiles typelist of profiles for the max boundary along the axis
                 *
                 * @param sourceTimeIteration time iteration at which the source incident B field
                 *                            (not the target E field!) values will be calculated
                 */
                template<uint32_t T_axis, typename T_MinProfiles, typename T_MaxProfiles>
                void updateE(float_X const sourceTimeIteration)
                {
                    auto parameters = detail::Parameters<T_axis>{cellDescription};
                    parameters.totalPositionMinBorder = totalPositionMinBorder;
                    parameters.totalPositionMaxBorder = totalPositionMaxBorder;
                    parameters.direction = 1.0_X;
                    parameters.sourceTimeIteration = sourceTimeIteration;
                    parameters.timeIncrement = maxwellSolver::getTimeStep();
                    meta::ForEach<T_MinProfiles, ApplyUpdateE<boost::mpl::_1>> applyMinProfiles;
                    applyMinProfiles(parameters);
                    parameters.direction = -1.0_X;
                    meta::ForEach<T_MaxProfiles, ApplyUpdateE<boost::mpl::_1>> applyMaxProfiles;
                    applyMaxProfiles(parameters);
                }

                /** Functor to apply update E for the given particular profile (not a typelist), axis and direction
                 *
                 * @tparam T_Profile incident field profile for the chosen part of the Huygens surface
                 */
                template<typename T_Profile>
                struct ApplyUpdateE
                {
                    /** Call update E with the given parameters
                     *
                     * @tparam T_Parameters parameters type
                     *
                     * @param parameters parameters
                     */
                    template<typename T_Parameters>
                    HINLINE void operator()(T_Parameters const& parameters) const
                    {
                        using Functor = detail::FunctorIncidentB<T_Profile>;
                        using Update = typename detail::UpdateE<Functor>;
                        constexpr bool extendTransversalHuygensSurface
                            = profiles::makePeriodicTransversalHuygensSurfaceContiguous<T_Profile>;
                        Update{}(parameters, extendTransversalHuygensSurface);
                    }
                };

                /** Apply contribution of the incident E field to the B field update by half a time step
                 *
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_MinProfiles typelist of profiles for the min boundary along the axis
                 * @tparam T_MaxProfiles typelist of profiles for the max boundary along the axis
                 *
                 * @param sourceTimeIteration time iteration at which the source incident E field
                 *                            (not the target B field!) values will be calculated
                 */
                template<uint32_t T_axis, typename T_MinProfiles, typename T_MaxProfiles>
                void updateBHalf(float_X const sourceTimeIteration)
                {
                    auto parameters = detail::Parameters<T_axis>{cellDescription};
                    parameters.totalPositionMinBorder = totalPositionMinBorder;
                    parameters.totalPositionMaxBorder = totalPositionMaxBorder;
                    parameters.direction = 1.0_X;
                    parameters.sourceTimeIteration = sourceTimeIteration;
                    parameters.timeIncrement = 0.5_X * maxwellSolver::getTimeStep();
                    meta::ForEach<T_MinProfiles, ApplyUpdateB<boost::mpl::_1>> applyMinProfiles;
                    applyMinProfiles(parameters);
                    parameters.direction = -1.0_X;
                    meta::ForEach<T_MaxProfiles, ApplyUpdateB<boost::mpl::_1>> applyMaxProfiles;
                    applyMaxProfiles(parameters);
                }

                /** Functor to apply update B for the given particular profile (not a typelist), axis and direction
                 *
                 * @tparam T_Profile incident field profile for the chosen part of the Huygens surface
                 */
                template<typename T_Profile>
                struct ApplyUpdateB
                {
                    /** Call update B with the given parameters
                     *
                     * @tparam T_Parameters parameters type
                     *
                     * @param parameters parameters
                     */
                    template<typename T_Parameters>
                    HINLINE void operator()(T_Parameters const& parameters) const
                    {
                        using Functor = detail::FunctorIncidentE<T_Profile>;
                        using Update = typename detail::UpdateB<Functor>;
                        constexpr bool extendTransversalHuygensSurface
                            = profiles::makePeriodicTransversalHuygensSurfaceContiguous<T_Profile>;
                        Update{}(parameters, extendTransversalHuygensSurface);
                    }
                };

                /** Typelists of profiles to be used by implementation
                 *
                 * Make aliases to user-provided types for decoupling and uniformity of 2d and 3d cases.
                 * Always convert to typelist for uniform handling.
                 *
                 * @{
                 */

                using XMinProfiles = pmacc::MakeSeq_t<XMin>;
                using XMaxProfiles = pmacc::MakeSeq_t<XMax>;
                using YMinProfiles = pmacc::MakeSeq_t<YMin>;
                using YMaxProfiles = pmacc::MakeSeq_t<YMax>;
                using ZMinProfiles = pmacc::MakeSeq_t<std::conditional_t<simDim == 3, ZMin, profiles::None>>;
                using ZMaxProfiles = pmacc::MakeSeq_t<std::conditional_t<simDim == 3, ZMax, profiles::None>>;

                /** @} */

                /** Total position of the Huygens surface for min and max borders
                 *
                 * It is offset from the origin of the user coordinate system, in cells.
                 * The Huygens surface is additionally offset by 0.75 cells inwards from this position.
                 *
                 * @{
                 */
                pmacc::DataSpace<simDim> totalPositionMinBorder;
                pmacc::DataSpace<simDim> totalPositionMaxBorder;
                /** @} */

                //! Cell description for kernels
                MappingDesc const cellDescription;
            };

        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
