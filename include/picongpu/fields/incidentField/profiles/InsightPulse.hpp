/* Copyright 2024 Fabia Dietrich
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/incidentField/Functors.hpp"
#include "picongpu/fields/incidentField/Traits.hpp"
#include "picongpu/fields/incidentField/profiles/InsightPulse.def"

#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <openPMD/openPMD.hpp>


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
                    /** Unitless InsightPulse parameters
                     *
                     * These parameters do not inherit from BaseParam, since some of them
                     * are unneccesary for this Laser implementation. For the remaining
                     * (necessary) base parameters, the calculations/functions/asserts are
                     * partly copied from there.
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct InsightPulseUnitless : public T_Params
                    {
                        //! User SI parameters
                        using Params = T_Params;

                        //! Check that simulation dimension = 3, since the recorded data is 3D aswell
                        PMACC_CASSERT_MSG(_error_simDim_has_to_be_3, simDim == 3u);

                        //! Unit propagation direction vector in 3d
                        static constexpr float_X DIR_X = static_cast<float_X>(Params::DIRECTION_X);
                        static constexpr float_X DIR_Y = static_cast<float_X>(Params::DIRECTION_Y);
                        static constexpr float_X DIR_Z = static_cast<float_X>(Params::DIRECTION_Z);

                        // Check that direction is normalized
                        static constexpr float_X dirNorm2 = DIR_X * DIR_X + DIR_Y * DIR_Y + DIR_Z * DIR_Z;
                        PMACC_CASSERT_MSG(
                            _error_laser_direction_vector_must_be_unit____check_your_incidentField_param_file,
                            (dirNorm2 > 0.9999) and (dirNorm2 < 1.0001));

                        // Check that just one axis is used as propagation direction
                        PMACC_CASSERT_MSG(
                            _error_laser_direction_vector_must_be_limited_to_one_axis____check_your_incidentField_param_file,
                            (DIR_X * DIR_X > 0.9999) and (DIR_X * DIR_X < 1.0001)
                                or (DIR_Y * DIR_Y > 0.9999) and (DIR_Y * DIR_Y < 1.0001)
                                or (DIR_Z * DIR_Z > 0.9999) and (DIR_Z * DIR_Z < 1.0001));

                        //! Unit polarization direction vector
                        static constexpr float_X POL_DIR_X = static_cast<float_X>(Params::POLARISATION_DIRECTION_X);
                        static constexpr float_X POL_DIR_Y = static_cast<float_X>(Params::POLARISATION_DIRECTION_Y);
                        static constexpr float_X POL_DIR_Z = static_cast<float_X>(Params::POLARISATION_DIRECTION_Z);

                        // Check that polarization direction is normalized
                        static constexpr float_X polDirNorm2
                            = POL_DIR_X * POL_DIR_X + POL_DIR_Y * POL_DIR_Y + POL_DIR_Z * POL_DIR_Z;
                        PMACC_CASSERT_MSG(
                            _error_laser_polarization_direction_vector_must_be_unit____check_your_incidentField_param_file,
                            (polDirNorm2 > 0.9999) && (polDirNorm2 < 1.0001));

                        // Check that just one axis is used as polarisation direction
                        PMACC_CASSERT_MSG(
                            _error_laser_direction_vector_must_be_limited_to_one_axis____check_your_incidentField_param_file,
                            (POL_DIR_X * POL_DIR_X > 0.9999) and (POL_DIR_X * POL_DIR_X < 1.0001)
                                or (POL_DIR_Y * POL_DIR_Y > 0.9999) and (POL_DIR_Y * POL_DIR_Y < 1.0001)
                                or (POL_DIR_Z * POL_DIR_Z > 0.9999) and (POL_DIR_Z * POL_DIR_Z < 1.0001));

                        // Check that polarization direction is orthogonal to propagation direction
                        static constexpr float_X dotPropagationPolarization
                            = DIR_X * POL_DIR_X + DIR_Y * POL_DIR_Y + DIR_Z * POL_DIR_Z;
                        PMACC_CASSERT_MSG(
                            _error_laser_polarization_direction_vector_must_be_orthogonal_to_propagation_direction____check_your_incidentField_param_file,
                            (dotPropagationPolarization > -0.0001) && (dotPropagationPolarization < 0.0001));

                        /** Time delay
                         *
                         * This parameter is *not* optional, as it is in other Laser implementations.
                         *
                         * unit: UNIT_TIME
                         */
                        static constexpr float_X TIME_DELAY = static_cast<float_X>(Params::TIME_DELAY_SI / UNIT_TIME);
                        PMACC_CASSERT_MSG(
                            _error_laser_time_delay_must_be_positive____check_your_incidentField_param_file,
                            (TIME_DELAY >= 0.0));

                        // check OpenPMD propagation direction
                        PMACC_CASSERT_MSG(
                            _error_propagationAxisOpenPMD_is_not_valid____check_your_parameters,
                            (Params::propagationAxisOpenPMD == "x" or Params::propagationAxisOpenPMD == "y"
                             or Params::propagationAxisOpenPMD == "z"));

                        // check OpenPMD polarisation direction
                        PMACC_CASSERT_MSG(
                            _error_polarisationAxisOpenPMD_is_not_valid____check_your_parameters,
                            (Params::polarisationAxisOpenPMD == "x" or Params::polarisationAxisOpenPMD == "y"
                             or Params::polarisationAxisOpenPMD == "z"));

                        PMACC_CASSERT_MSG(
                            _error_propagationAxisOpenPMD_and_polarisationAxisOpenPMD_have_to_be_different_____check_your_parameters,
                            (Params::polarisationAxisOpenPMD != Params::propagationAxisOpenPMD));
                    };

                    template<typename T_Params>
                    struct InsightPulseFunctorIncidentE;

                    /** Singleton to load field data from OpenPMD to device
                     *
                     * The complete dataset will be loaded (equally) to all GPUs, aswell as
                     * the necessary attributes (extent, cell size, offset to simulation window).
                     *
                     * Right now, the data will be loaded at timestep 0, which means that the user
                     * has to **increase the reserved GPU memory** in memory.param, since otherwise
                     * the simulation will run into memory issues.
                     * Future improvements of this implementation should consider loading the data
                     * before the start of the simulation (before the particle memory gets allocated),
                     * since this would solve this bug.
                     *
                     * @tparam T_Params user parameters, providing filename etc.
                     */
                    template<typename T_Params>
                    struct OpenPMDdata : public InsightPulseUnitless<T_Params>
                    {
                        //! Unitless parameters type
                        using Params = InsightPulseUnitless<T_Params>;
                        using dataType = typename Params::dataType;

                        //! Insight Pulse E functor
                        using Functor = InsightPulseFunctorIncidentE<T_Params>;

                        //! HostDeviceBuffer to store E field data
                        std::shared_ptr<pmacc::HostDeviceBuffer<float_X, simDim>> bufferFieldData;

                        //! HostDeviceBuffer to store the necessary attributes
                        std::shared_ptr<pmacc::HostDeviceBuffer<float_X, 1>> bufferExtentOpenPMD;
                        std::shared_ptr<pmacc::HostDeviceBuffer<float_X, 1>> bufferCellSizeOpenPMD;
                        std::shared_ptr<pmacc::HostDeviceBuffer<float_X, 1>> bufferOffsetOpenPMD;

                        //! loading data to device
                        static OpenPMDdata& get()
                        {
                            static OpenPMDdata dataBuffers{};
                            return dataBuffers;
                        }

                    private:
                        OpenPMDdata()
                        {
                            /* Open a series (this does not read the dataset itself).
                             * This is MPI collective and so has to be done by all ranks.
                             */
                            auto& gc = Environment<simDim>::get().GridController();

                            auto series = ::openPMD::Series{
                                Params::filename,
                                ::openPMD::Access::READ_ONLY,
                                gc.getCommunicator().getMPIComm()};
                            ::openPMD::Mesh mesh = series.iterations[Params::iteration].meshes[Params::datasetEName];
                            // check data order
                            if(mesh.dataOrder() != ::openPMD::Mesh::DataOrder::C)
                                throw std::runtime_error(
                                    "Unsupported dataOrder in FromOpenPMD density dataset, only C is supported");
                            // get axis labels
                            auto const axisLabels = std::vector<std::string>{mesh.axisLabels()};

                            // aligning recorded field data according to user input propagation / polarization
                            // direction
                            float3_X const xyzAxisIndex{0.0_X, 1.0_X, 2.0_X};

                            // starting with aligning the second transversal direction, perp. to polarisation and
                            // propagation
                            DataSpace<simDim> aligningAxisIndex
                                = DataSpace<simDim>::create(static_cast<int>(pmacc::math::abs(pmacc::math::dot(
                                    xyzAxisIndex,
                                    pmacc::math::cross(Functor::getDirection(), Functor::getPolarisationVector())))));

                            // aligning propagation direction
                            int const propAxisIdx = static_cast<int>(
                                pmacc::math::abs(pmacc::math::dot(xyzAxisIndex, Functor::getDirection())));
                            auto it_prop
                                = std::find(axisLabels.begin(), axisLabels.end(), Params::propagationAxisOpenPMD);
                            if(it_prop != std::end(axisLabels))
                                aligningAxisIndex[std::distance(begin(axisLabels), it_prop)] = propAxisIdx;
                            else
                                throw std::runtime_error(
                                    "Error: could not find propagation axis "
                                    + std::string(Params::propagationAxisOpenPMD) + " in OpenPMD dataset");

                            // aligning polarisation direction
                            auto it_pola
                                = std::find(axisLabels.begin(), axisLabels.end(), Params::polarisationAxisOpenPMD);
                            if(it_pola != std::end(axisLabels))
                                aligningAxisIndex[std::distance(begin(axisLabels), it_pola)]
                                    = static_cast<int>(pmacc::math::abs(
                                        pmacc::math::dot(xyzAxisIndex, Functor::getPolarisationVector())));
                            else
                                throw std::runtime_error(
                                    "Could not find polarisation axis " + std::string(Params::polarisationAxisOpenPMD)
                                    + " in OpenPMD dataset");

                            ::openPMD::MeshRecordComponent meshRecord = mesh[Params::polarisationAxisOpenPMD];

                            //! necessary attributes
                            // Raw = not yet aligned
                            ::openPMD::Extent const extentRaw = meshRecord.getExtent();
                            auto const cellSizeRaw = mesh.gridSpacing<dataType>();

                            bufferExtentOpenPMD = std::make_shared<pmacc::HostDeviceBuffer<float_X, 1>>(simDim);
                            bufferCellSizeOpenPMD = std::make_shared<pmacc::HostDeviceBuffer<float_X, 1>>(simDim);
                            bufferOffsetOpenPMD = std::make_shared<pmacc::HostDeviceBuffer<float_X, 1>>(simDim);

                            auto dataBoxExtent = bufferExtentOpenPMD->getHostBuffer().getDataBox();
                            auto dataBoxCellSize = bufferCellSizeOpenPMD->getHostBuffer().getDataBox();
                            auto dataBoxOffset = bufferOffsetOpenPMD->getHostBuffer().getDataBox();

                            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                            const auto extentPIC(subGrid.getGlobalDomain().size);
                            DataSpace<simDim> extentOpenPMD;

                            for(uint32_t d = 0u; d < simDim; d++)
                            {
                                // axis alignment and type conversion
                                extentOpenPMD[aligningAxisIndex[d]] = static_cast<int>(extentRaw[d]);
                                dataBoxExtent(aligningAxisIndex[d]) = static_cast<float_X>(extentRaw[d]);
                                dataBoxCellSize(aligningAxisIndex[d])
                                    = static_cast<float_X>(cellSizeRaw[d] * mesh.gridUnitSI()) / UNIT_LENGTH;
                                dataBoxOffset(aligningAxisIndex[d]) = 0.5_X
                                    * (static_cast<float_X>(extentPIC[aligningAxisIndex[d]] - 1)
                                           * cellSize[aligningAxisIndex[d]]
                                       - (dataBoxExtent(aligningAxisIndex[d]) - 1.0_X)
                                           * dataBoxCellSize(aligningAxisIndex[d]));
                            }

                            // push attribute data to device
                            bufferExtentOpenPMD->hostToDevice();
                            bufferCellSizeOpenPMD->hostToDevice();
                            bufferOffsetOpenPMD->hostToDevice();
                            eventSystem::getTransactionEvent().waitForFinished();

                            // field data
                            bufferFieldData
                                = std::make_shared<pmacc::HostDeviceBuffer<float_X, simDim>>(extentOpenPMD);
                            auto fieldData = std::shared_ptr<dataType>{nullptr};
                            fieldData = meshRecord.loadChunk<dataType>();

                            // This is MPI collective and so has to be done by all ranks
                            series.flush();

                            auto const numElements = std::accumulate(
                                std::begin(extentRaw),
                                std::end(extentRaw),
                                1u,
                                std::multiplies<uint32_t>());

                            auto hostFieldDataBox = bufferFieldData->getHostBuffer().getDataBox();

                            // reshaping, aligning and type casting of recorded field data
                            for(uint32_t linearIdx = 0u; linearIdx < numElements; linearIdx++)
                            {
                                DataSpace<simDim> openPMDIdx;
                                auto tmpIndex = linearIdx;
                                for(int32_t d = simDim - 1; d >= 0; d--)
                                {
                                    openPMDIdx[aligningAxisIndex[d]] = tmpIndex % extentRaw[d];
                                    tmpIndex /= extentRaw[d];
                                }
                                hostFieldDataBox(openPMDIdx)
                                    = static_cast<float_X>(fieldData.get()[linearIdx] * meshRecord.unitSI())
                                    / UNIT_EFIELD;
                            }

                            /* If the transversal simulation window is smaller than the transversal DataBox extent,
                             * data at the chunk borders will be discarded. The maximum discarded value (relative to
                             * the maximum amplitude) will be logged. A possible improvement of this code could include
                             * storing just the necessary parts of the recorded field data.
                             */
                            dataType maxE(0.0);
                            dataType maxEDiscarded(0.0);
                            dataType valE, valRight, valLeft;
                            bool discard = false;
                            for(uint32_t d = 0u; d < simDim - 1; d++)
                            {
                                if(d != propAxisIdx and dataBoxOffset(d) < 0)
                                {
                                    for(uint32_t i = 0; i < extentOpenPMD[0]; i++)
                                    {
                                        for(uint32_t j = 0; j < extentOpenPMD[1]; j++)
                                        {
                                            for(uint32_t k = 0; k < extentOpenPMD[2]; k++)
                                            {
                                                // look for maximum amplitude value
                                                if(discard == false) // do this just the first time entering the loop
                                                {
                                                    valE = pmacc::math::abs(
                                                        hostFieldDataBox(DataSpace<simDim>(i, j, k)));
                                                    if(valE > maxE)
                                                        maxE = valE;
                                                }
                                                // look for maximum discarded amplitude value
                                                if(d == 0
                                                   and i <= static_cast<int>(
                                                           pmacc::math::abs(dataBoxOffset(d)) / dataBoxCellSize(d)))
                                                {
                                                    valLeft = pmacc::math::abs(
                                                        hostFieldDataBox(DataSpace<simDim>(i, j, k)));
                                                    if(valLeft > maxEDiscarded) // left discarded area
                                                        maxEDiscarded = valLeft;
                                                    valRight = pmacc::math::abs(hostFieldDataBox(
                                                        DataSpace<simDim>(extentOpenPMD[d] - 1 - i, j, k)));
                                                    if(valRight > maxEDiscarded) // right discarded area
                                                        maxEDiscarded = valRight;
                                                }
                                                if(d == 1
                                                   and j <= static_cast<int>(
                                                           pmacc::math::abs(dataBoxOffset(d)) / dataBoxCellSize(d)))
                                                {
                                                    valLeft = pmacc::math::abs(
                                                        hostFieldDataBox(DataSpace<simDim>(i, j, k)));
                                                    if(valLeft > maxEDiscarded) // left discarded area
                                                        maxEDiscarded = valLeft;
                                                    valRight = pmacc::math::abs(hostFieldDataBox(
                                                        DataSpace<simDim>(i, extentOpenPMD[d] - 1 - j, k)));
                                                    if(valRight > maxEDiscarded) // right discarded area
                                                        maxEDiscarded = valRight;
                                                }
                                                if(d == 2
                                                   and k <= static_cast<int>(
                                                           pmacc::math::abs(dataBoxOffset(d)) / dataBoxCellSize(d)))
                                                {
                                                    valLeft = pmacc::math::abs(
                                                        hostFieldDataBox(DataSpace<simDim>(i, j, k)));
                                                    if(valLeft > maxEDiscarded) // left discarded area
                                                        maxEDiscarded = valLeft;
                                                    valRight = pmacc::math::abs(hostFieldDataBox(
                                                        DataSpace<simDim>(i, j, extentOpenPMD[d] - 1 - k)));
                                                    if(valRight > maxEDiscarded) // right discarded area
                                                        maxEDiscarded = valRight;
                                                }
                                            } // k
                                        } // j
                                    } // i
                                    discard = true;
                                }
                            } // d

                            // This message is shown for every used device, but 1x would be sufficient (aswell as the
                            // loop before)
                            if(discard == true)
                                log<picLog::PHYSICS>("Warning: Transversal simulation window extent is smaller than "
                                                     "measured data, discarding data at the border.\nMax. discarded "
                                                     "amplitude relative to max. measured amplitude: %1% ")
                                    % (maxEDiscarded / maxE);

                            /* Push field data to device
                             * Every used device will now store the whole field data chunk. Since this consumes a lot
                             * of memory, a possible improvement of this code could be to push only those two time
                             * slices to the device, which are necessary for the current time step.
                             */
                            bufferFieldData->hostToDevice();
                            eventSystem::getTransactionEvent().waitForFinished();
                        }
                    };

                    /** InsightPulse incident E functor
                     *
                     * @tparam T_Params parameters
                     */
                    template<typename T_Params>
                    struct InsightPulseFunctorIncidentE : public InsightPulseUnitless<T_Params>
                    {
                        //! Unitless parameters type
                        using Unitless = InsightPulseUnitless<T_Params>;

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE InsightPulseFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
                            : timeOriginPIC(currentStep * DELTA_T)
                        {
                            // load data at timestep 0
                            auto& openPMDdata = OpenPMDdata<T_Params>::get();

                            // get field data
                            fieldDataBox = openPMDdata.bufferFieldData->getDeviceBuffer().getDataBox();

                            // get field data attributes
                            extentOpenPMDdataBox = openPMDdata.bufferExtentOpenPMD->getDeviceBuffer().getDataBox();
                            cellSizeOpenPMDdataBox = openPMDdata.bufferCellSizeOpenPMD->getDeviceBuffer().getDataBox();
                            offsetOpenPMDdataBox = openPMDdata.bufferOffsetOpenPMD->getDeviceBuffer().getDataBox();
                        }

                        /** Read incident field E value for the given position and time step
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @return incident field E value in internal units
                         */
                        HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
                        {
                            return getPolarisationVector() * getValueE(totalCellIdx);
                        }

                        //! Get a unit vector with linear E polarization
                        HDINLINE static constexpr float3_X getPolarisationVector()
                        {
                            return float3_X(Unitless::POL_DIR_X, Unitless::POL_DIR_Y, Unitless::POL_DIR_Z);
                        }

                        //! Get a 3-dimensional unit direction vector
                        HDINLINE static constexpr float3_X getDirection()
                        {
                            return float3_X(Unitless::DIR_X, Unitless::DIR_Y, Unitless::DIR_Z);
                        }

                    private:
                        /** Get value of E field for the given position
                         * Linear interpolation of measured field data, which is aligned centered at the chosen
                         * incident field plane. If the simulation window extent is greater than the field data chunk,
                         * zero will be returned. In the other case, the field data chunk will be (transversally)
                         * cropped.
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         */
                        HDINLINE float_X getValueE(floatD_X const& totalCellIdx) const
                        {
                            auto const posPIC = totalCellIdx * cellSize; // position in simulation volume

                            // find the axis indices
                            float3_X const xyzAxisIdxPlusOne(
                                1.0_X,
                                2.0_X,
                                3.0_X); // axis indices + 1 (needed for the sign)
                            int const propAxisIdx = static_cast<int>(pmacc::math::abs(
                                                        pmacc::math::dot(xyzAxisIdxPlusOne, getDirection())))
                                - 1;
                            int const polAxisIdxPlusOne
                                = static_cast<int>(pmacc::math::dot(xyzAxisIdxPlusOne, getPolarisationVector()));
                            int const transvAxisIdxPlusOne = static_cast<int>(pmacc::math::dot(
                                xyzAxisIdxPlusOne,
                                pmacc::math::cross(getDirection(), getPolarisationVector())));

                            // check whether the system is right handed, i.e. direction x polarisation > 0
                            bool rh = true; // > 0
                            if(Unitless::propagationAxisOpenPMD == "x" and Unitless::polarisationAxisOpenPMD == "z"
                               or Unitless::propagationAxisOpenPMD == "y" and Unitless::polarisationAxisOpenPMD == "x"
                               or Unitless::propagationAxisOpenPMD == "z" and Unitless::polarisationAxisOpenPMD == "y")
                                rh = false; // < 0

                            // find the index in field data which is closest to totalCellIdx and timeOriginPIC
                            float3_X idxClosestRaw; // raw = not yet rounded to integers

                            for(uint32_t d = 0u; d < simDim; d++)
                            {
                                // return zero if transversal simulation window or simulation timestep exceeds chunk
                                // extent
                                if(d != propAxisIdx
                                       and (posPIC[d] < offsetOpenPMDdataBox(d) or posPIC[d] > offsetOpenPMDdataBox(d) + (extentOpenPMDdataBox(d) - 1.0_X) * cellSizeOpenPMDdataBox(d))
                                   or (timeOriginPIC - Unitless::TIME_DELAY) < 0.0_X
                                   or (timeOriginPIC - Unitless::TIME_DELAY) > (extentOpenPMDdataBox(d) - 1.0_X)
                                           * cellSizeOpenPMDdataBox(d) / SPEED_OF_LIGHT)
                                    return 0.0_X;
                                else
                                {
                                    if(d == pmacc::math::abs(polAxisIdxPlusOne) - 1 and polAxisIdxPlusOne < 0
                                       or d == pmacc::math::abs(transvAxisIdxPlusOne) - 1
                                           and (transvAxisIdxPlusOne < 0 and rh == true or transvAxisIdxPlusOne > 0 and rh == false))
                                        // invert direction
                                        idxClosestRaw[d] = extentOpenPMDdataBox(d) - 1.0_X
                                            - (posPIC[d] - offsetOpenPMDdataBox(d)) / cellSizeOpenPMDdataBox(d);
                                    else // keep original direction
                                        idxClosestRaw[d]
                                            = (posPIC[d] - offsetOpenPMDdataBox(d)) / cellSizeOpenPMDdataBox(d);
                                }
                            }

                            // correct longitudinal index
                            idxClosestRaw[propAxisIdx] = extentOpenPMDdataBox(propAxisIdx) - 1.0_X
                                - (timeOriginPIC - Unitless::TIME_DELAY) / cellSizeOpenPMDdataBox(propAxisIdx)
                                    * SPEED_OF_LIGHT;

                            DataSpace<simDim> const idxClosest = static_cast<pmacc::math::Vector<int, simDim>>(
                                idxClosestRaw + floatD_X::create(0.5_X));
                            // the other 7 nearest neighbour indices still have to be found
                            DataSpace<simDim> idxShift; // shift to the other nearest neighbour indices
                            DataSpace<simDim> idxNext; // nearest neighbour index
                            floatD_X weight;
                            for(uint32_t d = 0u; d < simDim; d++)
                            {
                                if(idxClosest[d] == 0) // to avoid border problems
                                    idxShift[d] = 1;
                                else if(idxClosestRaw[d] - static_cast<float_X>(idxClosest[d]) <= 0.0_X)
                                    idxShift[d] = -1;
                                else
                                    idxShift[d] = 1;

                                weight[d] = pmacc::math::abs(static_cast<float_X>(idxClosest[d]) - idxClosestRaw[d]);
                            }

                            // linear interpolation routine
                            float_X interpE = 0.0_X;

                            interpE
                                += fieldDataBox(idxClosest) * (floatD_X::create(1.0_X) - weight).productOfComponents();
                            for(uint32_t d = 0u; d < simDim; d++)
                            {
                                idxNext = idxClosest;
                                floatD_X ones = floatD_X::create(1.0_X);
                                idxNext[d] = idxClosest[d] + idxShift[d];
                                ones[d] = 0.0_X;
                                interpE -= fieldDataBox(idxNext) * (ones - weight).productOfComponents();
                                if(d == 2)
                                {
                                    idxNext[0] = idxClosest[0] + idxShift[0];
                                    ones[0] = 0.0_X;
                                }
                                else
                                {
                                    idxNext[d + 1] = idxClosest[d + 1] + idxShift[d + 1];
                                    ones[d + 1] = 0.0_X;
                                }
                                interpE += fieldDataBox(idxNext) * (ones - weight).productOfComponents();
                            }
                            interpE += fieldDataBox(idxClosest + idxShift) * weight.productOfComponents();

                            return interpE;
                        } // getValueE

                    protected:
                        float_X const timeOriginPIC; // current time at incident plane
                        PMACC_ALIGN(fieldDataBox, typename pmacc::Buffer<float_X, simDim>::DataBoxType);
                        typename pmacc::Buffer<float_X, 1>::DataBoxType extentOpenPMDdataBox;
                        typename pmacc::Buffer<float_X, 1>::DataBoxType cellSizeOpenPMDdataBox;
                        typename pmacc::Buffer<float_X, 1>::DataBoxType offsetOpenPMDdataBox;

                    }; // InsightPulseFunctorIncidentE
                } // namespace detail

                template<typename T_Params>
                struct InsightPulse
                {
                    //! Get text name of the incident field profile
                    static HINLINE std::string getName()
                    {
                        return "InsightPulse";
                    }
                };
            } // namespace profiles

            namespace detail
            {
                /** Get type of incident field E functor for the Insight laser profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentE<profiles::InsightPulse<T_Params>>
                {
                    using type = profiles::detail::InsightPulseFunctorIncidentE<T_Params>;
                };

                /** Get type of incident field B functor for the Insight laser profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentB<profiles::InsightPulse<T_Params>>
                {
                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::InsightPulse<T_Params>>::type>;
                };

            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
