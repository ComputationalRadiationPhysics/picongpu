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

#if(ENABLE_OPENPMD == 1) && (SIMDIM == DIM3)

#    pragma once

#    include "picongpu/simulation_defines.hpp"

#    include "picongpu/fields/incidentField/Functors.hpp"
#    include "picongpu/fields/incidentField/Traits.hpp"
#    include "picongpu/fields/incidentField/profiles/FromOpenPMDPulse.def"

#    include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#    include <algorithm>
#    include <array>
#    include <cmath>
#    include <cstdint>
#    include <limits>
#    include <memory>
#    include <string>
#    include <type_traits>
#    include <vector>

#    include <openPMD/openPMD.hpp>

/* REFACTORING IDEAS FOR THIS INCIDENT FIELD PROFILE
 * -------------------------------------------------
 * - make time delay parameter optional
 * - load openPMD file (= call the corresponding singelton) or initialize the
 *   Laser once before timestep 0 (before particle memory allocation)
 * - load just the necessary parts of the measured data if the tranversal
 *   simulation window extent is smaller than the transversal field chunk size
 * - allow diagonal laser propagation instead of just parallel to the axes
 * - every used device will store the whole field data chunk, which consumes
 *   quite some memory. Instead, one could push only those two time slices to
 *   the device which are necessary for the current time step.
 * - get rid of the 'wrong' transformation from time to space (z = c*t) of the
 *   longitudinal axis by using several iterations inside the openPMD file
 *   instead of just one
 */

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
                    /** Unitless FromOpenPMDPulse parameters
                     *
                     * These parameters do not inherit from BaseParam, since some of them
                     * are unneccesary for this Laser implementation. For the remaining
                     * (necessary) base parameters, the calculations/functions/asserts are
                     * partly copied from there.
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct FromOpenPMDPulseUnitless : public T_Params
                    {
                        //! User SI parameters
                        using Params = T_Params;

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

                        //! Unit polarisation direction vector
                        static constexpr float_X POL_DIR_X = static_cast<float_X>(Params::POLARISATION_DIRECTION_X);
                        static constexpr float_X POL_DIR_Y = static_cast<float_X>(Params::POLARISATION_DIRECTION_Y);
                        static constexpr float_X POL_DIR_Z = static_cast<float_X>(Params::POLARISATION_DIRECTION_Z);

                        // Check that polarisation direction is normalized
                        static constexpr float_X polDirNorm2
                            = POL_DIR_X * POL_DIR_X + POL_DIR_Y * POL_DIR_Y + POL_DIR_Z * POL_DIR_Z;
                        PMACC_CASSERT_MSG(
                            _error_laser_polarisation_direction_vector_must_be_unit____check_your_incidentField_param_file,
                            (polDirNorm2 > 0.9999) && (polDirNorm2 < 1.0001));

                        // Check that just one axis is used as polarisation direction
                        PMACC_CASSERT_MSG(
                            _error_laser_direction_vector_must_be_limited_to_one_axis____check_your_incidentField_param_file,
                            (POL_DIR_X * POL_DIR_X > 0.9999) and (POL_DIR_X * POL_DIR_X < 1.0001)
                                or (POL_DIR_Y * POL_DIR_Y > 0.9999) and (POL_DIR_Y * POL_DIR_Y < 1.0001)
                                or (POL_DIR_Z * POL_DIR_Z > 0.9999) and (POL_DIR_Z * POL_DIR_Z < 1.0001));

                        // Check that polarisation direction is orthogonal to propagation direction
                        static constexpr float_X dotPropagationPolarisation
                            = DIR_X * POL_DIR_X + DIR_Y * POL_DIR_Y + DIR_Z * POL_DIR_Z;
                        PMACC_CASSERT_MSG(
                            _error_laser_polarisation_direction_vector_must_be_orthogonal_to_propagation_direction____check_your_incidentField_param_file,
                            (dotPropagationPolarisation > -0.0001) && (dotPropagationPolarisation < 0.0001));

                        /** Time delay
                         *
                         * This parameter is *not* optional, as it is in other Laser implementations.
                         *
                         * unit:  sim.unit.time()
                         */
                        static constexpr float_X TIME_DELAY
                            = static_cast<float_X>(Params::TIME_DELAY_SI / sim.unit.time());
                        PMACC_CASSERT_MSG(
                            _error_laser_time_delay_must_be_positive____check_your_incidentField_param_file,
                            (TIME_DELAY >= 0.0));

                        // check openPMD propagation direction
                        PMACC_CASSERT_MSG(
                            _error_propagationAxisOpenPMD_is_not_valid____check_your_parameters,
                            (Params::propagationAxisOpenPMD == "x" or Params::propagationAxisOpenPMD == "y"
                             or Params::propagationAxisOpenPMD == "z"));

                        // check openPMD polarisation direction
                        PMACC_CASSERT_MSG(
                            _error_polarisationAxisOpenPMD_is_not_valid____check_your_parameters,
                            (Params::polarisationAxisOpenPMD == "x" or Params::polarisationAxisOpenPMD == "y"
                             or Params::polarisationAxisOpenPMD == "z"));

                        PMACC_CASSERT_MSG(
                            _error_propagationAxisOpenPMD_and_polarisationAxisOpenPMD_have_to_be_different_____check_your_parameters,
                            (Params::polarisationAxisOpenPMD != Params::propagationAxisOpenPMD));
                    };

                    template<typename T_Params>
                    struct FromOpenPMDPulseFunctorIncidentE;

                    /** Singleton to load field data from openPMD to device
                     *
                     * The complete dataset will be loaded (equally) to all GPUs, as well as
                     * the necessary attributes (extent, cell size, offset to simulation window).
                     *
                     * Right now, the data will be loaded at timestep 0, which means that the user
                     * has to **increase the reserved GPU memory** in memory.param, since otherwise
                     * the simulation will run into memory issues.
                     *
                     * @tparam T_Params user parameters, providing filename etc.
                     */
                    template<typename T_Params>
                    struct OpenPMDdata : public FromOpenPMDPulseUnitless<T_Params>
                    {
                        //! Unitless parameters type
                        using Params = FromOpenPMDPulseUnitless<T_Params>;
                        using dataType = typename Params::dataType;

                        //! FromOpenPMD pulse E functor
                        using Functor = FromOpenPMDPulseFunctorIncidentE<T_Params>;

                        //! HostDeviceBuffer to store E field data
                        std::shared_ptr<pmacc::HostDeviceBuffer<float_X, 3u>> bufferFieldData;

                        //! HostDeviceBuffer to store the necessary attributes
                        std::shared_ptr<pmacc::HostDeviceBuffer<float_X, 1u>> bufferExtentOpenPMD;
                        std::shared_ptr<pmacc::HostDeviceBuffer<float_X, 1u>> bufferCellSizeOpenPMD;
                        std::shared_ptr<pmacc::HostDeviceBuffer<float_X, 1u>> bufferOffsetOpenPMD;

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
                                    "Unsupported dataOrder in openPMD E-field dataset, only C is supported");

                            /* Now we align the recorded field data according to the user input
                             * = rotation into the internal coordinate system
                             */
                            auto const axisLabels = std::vector<std::string>{mesh.axisLabels()};
                            DataSpace<3u> const internalAxisIndex = Functor::getInternalAxisIndex();

                            // start with the second transversal direction
                            DataSpace<3u> aligningAxisIndex = DataSpace<3u>::create(internalAxisIndex[2]);

                            // go on with the propagation direction
                            auto it_prop
                                = std::find(axisLabels.begin(), axisLabels.end(), Params::propagationAxisOpenPMD);
                            if(it_prop != std::end(axisLabels))
                                aligningAxisIndex[std::distance(begin(axisLabels), it_prop)] = internalAxisIndex[0];
                            else
                                throw std::runtime_error(
                                    "Error: could not find propagation axis "
                                    + std::string(Params::propagationAxisOpenPMD) + " in OpenPMD dataset");

                            // align the polarisation direction
                            auto it_pola
                                = std::find(axisLabels.begin(), axisLabels.end(), Params::polarisationAxisOpenPMD);
                            if(it_pola != std::end(axisLabels))
                                aligningAxisIndex[std::distance(begin(axisLabels), it_pola)] = internalAxisIndex[1];
                            else
                                throw std::runtime_error(
                                    "Could not find polarisation axis " + std::string(Params::polarisationAxisOpenPMD)
                                    + " in OpenPMD dataset");

                            ::openPMD::MeshRecordComponent meshRecord = mesh[Params::polarisationAxisOpenPMD];

                            //! necessary attributes
                            // Raw = not yet aligned
                            ::openPMD::Extent const extentRaw = meshRecord.getExtent();
                            auto const cellSizeRaw = mesh.gridSpacing<dataType>();

                            bufferExtentOpenPMD = std::make_shared<pmacc::HostDeviceBuffer<float_X, 1u>>(3u);
                            bufferCellSizeOpenPMD = std::make_shared<pmacc::HostDeviceBuffer<float_X, 1u>>(3u);
                            bufferOffsetOpenPMD = std::make_shared<pmacc::HostDeviceBuffer<float_X, 1u>>(3u);

                            auto dataBoxExtent = bufferExtentOpenPMD->getHostBuffer().getDataBox();
                            auto dataBoxCellSize = bufferCellSizeOpenPMD->getHostBuffer().getDataBox();
                            auto dataBoxOffset = bufferOffsetOpenPMD->getHostBuffer().getDataBox();

                            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                            const auto extentPIC(subGrid.getGlobalDomain().size);
                            DataSpace<3u> extentOpenPMD;

                            for(uint32_t d = 0u; d < 3u; d++)
                            {
                                // axis alignment and type conversion
                                extentOpenPMD[aligningAxisIndex[d]] = static_cast<int>(extentRaw[d]);
                                dataBoxExtent(aligningAxisIndex[d]) = static_cast<float_X>(extentRaw[d]);
                                dataBoxCellSize(aligningAxisIndex[d])
                                    = static_cast<float_X>(cellSizeRaw[d] * mesh.gridUnitSI()) / sim.unit.length();
                                dataBoxOffset(aligningAxisIndex[d]) = 0.5_X
                                    * (static_cast<float_X>(extentPIC[aligningAxisIndex[d]] - 1)
                                           * sim.pic.getCellSize()[aligningAxisIndex[d]]
                                       - (dataBoxExtent(aligningAxisIndex[d]) - 1.0_X)
                                           * dataBoxCellSize(aligningAxisIndex[d]));
                            }

                            // push attribute data to device
                            bufferExtentOpenPMD->hostToDevice();
                            bufferCellSizeOpenPMD->hostToDevice();
                            bufferOffsetOpenPMD->hostToDevice();
                            eventSystem::getTransactionEvent().waitForFinished();

                            // field data
                            bufferFieldData = std::make_shared<pmacc::HostDeviceBuffer<float_X, 3u>>(extentOpenPMD);
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
                                DataSpace<3u> openPMDIdx;
                                auto tmpIndex = linearIdx;
                                for(int32_t d = 2u; d >= 0; d--)
                                {
                                    openPMDIdx[aligningAxisIndex[d]] = tmpIndex % extentRaw[d];
                                    tmpIndex /= extentRaw[d];
                                }
                                hostFieldDataBox(openPMDIdx)
                                    = static_cast<float_X>(fieldData.get()[linearIdx] * meshRecord.unitSI())
                                    / sim.unit.eField();
                            }

                            /* If the transversal simulation window is smaller than the transversal DataBox extent,
                             * data at the chunk borders will be discarded. The maximum discarded value (relative to
                             * the maximum amplitude) will be logged.
                             */
                            dataType maxE(0.0);
                            dataType maxEDiscarded(0.0);
                            dataType valE, valRight, valLeft;
                            bool discard = false;
                            for(uint32_t d = 0u; d < 3u; d++)
                            {
                                if(d != internalAxisIndex[0] and dataBoxOffset(d) < 0)
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
                                                    valE = pmacc::math::abs(hostFieldDataBox(DataSpace<3u>(i, j, k)));
                                                    if(valE > maxE)
                                                        maxE = valE;
                                                }
                                                // look for maximum discarded amplitude value
                                                if(d == 0
                                                   and i <= static_cast<int>(
                                                           pmacc::math::abs(dataBoxOffset(d)) / dataBoxCellSize(d)))
                                                {
                                                    valLeft
                                                        = pmacc::math::abs(hostFieldDataBox(DataSpace<3u>(i, j, k)));
                                                    if(valLeft > maxEDiscarded) // left discarded area
                                                        maxEDiscarded = valLeft;
                                                    valRight = pmacc::math::abs(hostFieldDataBox(
                                                        DataSpace<3u>(extentOpenPMD[d] - 1 - i, j, k)));
                                                    if(valRight > maxEDiscarded) // right discarded area
                                                        maxEDiscarded = valRight;
                                                }
                                                if(d == 1
                                                   and j <= static_cast<int>(
                                                           pmacc::math::abs(dataBoxOffset(d)) / dataBoxCellSize(d)))
                                                {
                                                    valLeft
                                                        = pmacc::math::abs(hostFieldDataBox(DataSpace<3u>(i, j, k)));
                                                    if(valLeft > maxEDiscarded) // left discarded area
                                                        maxEDiscarded = valLeft;
                                                    valRight = pmacc::math::abs(hostFieldDataBox(
                                                        DataSpace<3u>(i, extentOpenPMD[d] - 1 - j, k)));
                                                    if(valRight > maxEDiscarded) // right discarded area
                                                        maxEDiscarded = valRight;
                                                }
                                                if(d == 2
                                                   and k <= static_cast<int>(
                                                           pmacc::math::abs(dataBoxOffset(d)) / dataBoxCellSize(d)))
                                                {
                                                    valLeft
                                                        = pmacc::math::abs(hostFieldDataBox(DataSpace<3u>(i, j, k)));
                                                    if(valLeft > maxEDiscarded) // left discarded area
                                                        maxEDiscarded = valLeft;
                                                    valRight = pmacc::math::abs(hostFieldDataBox(
                                                        DataSpace<3u>(i, j, extentOpenPMD[d] - 1 - k)));
                                                    if(valRight > maxEDiscarded) // right discarded area
                                                        maxEDiscarded = valRight;
                                                }
                                            } // k
                                        } // j
                                    } // i
                                    discard = true;
                                }
                            } // d

                            if(discard == true)
                            {
                                // show the discard warning message just for the 0th rank
                                pmacc::GridController<simDim>& gc = pmacc::Environment<simDim>::get().GridController();
                                pmacc::CommunicatorMPI<simDim>& comm = gc.getCommunicator();
                                uint32_t rank = comm.getRank();

                                if(rank == 0)
                                    log<picLog::PHYSICS>(
                                        "Warning: Transversal simulation window extent is smaller than "
                                        "measured data, discarding data at the border.\nMax. discarded "
                                        "amplitude relative to max. measured amplitude: %1% ")
                                        % (maxEDiscarded / maxE);
                            }

                            //! Push field data to device
                            bufferFieldData->hostToDevice();
                            eventSystem::getTransactionEvent().waitForFinished();
                        } // OpenPMDdata
                    };

                    /** FromOpenPMDPulse incident E functor
                     *
                     * @tparam T_Params parameters
                     */
                    template<typename T_Params>
                    struct FromOpenPMDPulseFunctorIncidentE : public FromOpenPMDPulseUnitless<T_Params>
                    {
                        //! Unitless parameters type
                        using Unitless = FromOpenPMDPulseUnitless<T_Params>;

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE FromOpenPMDPulseFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
                            : timeOriginPIC(currentStep * sim.pic.getDt())
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

                        //! Get a unit vector with linear E polarisation
                        HDINLINE static constexpr float3_X getPolarisationVector()
                        {
                            return float3_X(Unitless::POL_DIR_X, Unitless::POL_DIR_Y, Unitless::POL_DIR_Z);
                        }

                        //! Get a 3-dimensional unit direction vector
                        HDINLINE static constexpr float3_X getDirection()
                        {
                            return float3_X(Unitless::DIR_X, Unitless::DIR_Y, Unitless::DIR_Z);
                        }

                        /** Get the internal axis indices w.r.t. (x, y, z)
                         * [0] propagation axis index (longitudinal)
                         * [1] polarisation axis index
                         * [2] remaining transversal axis index
                         */
                        HDINLINE static DataSpace<3u> getInternalAxisIndex()
                        {
                            float3_X const xyzAxisIndex{0.0_X, 1.0_X, 2.0_X};
                            DataSpace<3u> const internalAxisIndex{
                                static_cast<int>(pmacc::math::abs(pmacc::math::dot(xyzAxisIndex, getDirection()))),
                                static_cast<int>(
                                    pmacc::math::abs(pmacc::math::dot(xyzAxisIndex, getPolarisationVector()))),
                                static_cast<int>(pmacc::math::abs(pmacc::math::dot(
                                    xyzAxisIndex,
                                    pmacc::math::cross(getDirection(), getPolarisationVector()))))};

                            return internalAxisIndex;
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
                            auto const posPIC = totalCellIdx * sim.pic.getCellSize(); // position in simulation volume

                            DataSpace<3u> internalAxisIndex = getInternalAxisIndex();

                            // check whether we are outside the field chunk extent
                            for(uint32_t d = 0u; d < simDim; d++)
                            {
                                // return zero if transversal simulation window exceeds chunk extent
                                if(d != internalAxisIndex[0])
                                {
                                    if(posPIC[d] < offsetOpenPMDdataBox(d)
                                       or posPIC[d] > offsetOpenPMDdataBox(d)
                                               + (extentOpenPMDdataBox(d) - 1.0_X) * cellSizeOpenPMDdataBox(d))
                                        return 0.0_X;
                                }
                                // return zero if simulation timestep exceeds chunk extent
                                else // d == internalAxisIndex[0]
                                {
                                    if((timeOriginPIC - Unitless::TIME_DELAY) < 0.0_X
                                       or (timeOriginPIC - Unitless::TIME_DELAY) > (extentOpenPMDdataBox(d) - 1.0_X)
                                               * cellSizeOpenPMDdataBox(d) / SPEED_OF_LIGHT)
                                        return 0.0_X;
                                }
                            }

                            // check whether the (openPMD) system is right handed, i.e. direction x polarisation > 0
                            bool rh = true; // > 0
                            if(Unitless::propagationAxisOpenPMD == "x" and Unitless::polarisationAxisOpenPMD == "z"
                               or Unitless::propagationAxisOpenPMD == "y" and Unitless::polarisationAxisOpenPMD == "x"
                               or Unitless::propagationAxisOpenPMD == "z" and Unitless::polarisationAxisOpenPMD == "y")
                                rh = false; // < 0

                            // transversal axis directions (internal)
                            float_X const polAxisDirection = getPolarisationVector().sumOfComponents();
                            float_X const transvAxisDirection
                                = pmacc::math::cross(getDirection(), getPolarisationVector()).sumOfComponents();

                            // find the position in the field data chunk corresponding to totalCellIdx and
                            // timeOriginPIC
                            float3_X idxClosestRaw; // raw = not yet rounded to integers

                            for(uint32_t d = 0u; d < simDim; d++)
                            {
                                /* Now we check whether we have to invert the transversal (internal) axis directions
                                 * in order to keep the handiness of the (openPMD) system
                                 */
                                if(d != internalAxisIndex[0])
                                {
                                    if(d == internalAxisIndex[1] and polAxisDirection < 0
                                       or d == internalAxisIndex[2]
                                           and (transvAxisDirection < 0 and rh or transvAxisDirection > 0 and !rh))
                                    {
                                        // we have to invert
                                        idxClosestRaw[d] = extentOpenPMDdataBox(d) - 1.0_X
                                            - (posPIC[d] - offsetOpenPMDdataBox(d)) / cellSizeOpenPMDdataBox(d);
                                    }
                                    else
                                    {
                                        // we can keep the original direction
                                        idxClosestRaw[d]
                                            = (posPIC[d] - offsetOpenPMDdataBox(d)) / cellSizeOpenPMDdataBox(d);
                                    }
                                } // d != internalAxisIndex[0]

                                else // d == internalAxisIndex[0]
                                {
                                    // the longitudinal (time) axis is always inverted
                                    idxClosestRaw[d] = extentOpenPMDdataBox(d) - 1.0_X
                                        - (timeOriginPIC - Unitless::TIME_DELAY) / cellSizeOpenPMDdataBox(d)
                                            * SPEED_OF_LIGHT;
                                }
                            } // for(uint32_t d = 0u; d < simDim; d++)

                            // linear interpolation
                            return linInterpol(fieldDataBox, idxClosestRaw);


                        } // getValueE

                        /** Linear interpolation routine
                         *
                         * @param dataBox 3D (Device) data box containing the data to be interpolated.
                         *        The spacing between the values is assumed to be one and the origin is located in the
                         *        (lower left) corner, i.e. the scales describing the data box == the cell indices
                         * @param pos position at which to evaluate; has to be inside the data box.
                         *
                         * returns the interpolated value of dataBox at pos.
                         */
                        HDINLINE float_X linInterpol(
                            typename pmacc::Buffer<float_X, 3u>::DataBoxType const& dataBox,
                            float3_X const& pos) const
                        {
                            // find the index in the data box which is nearest to pos
                            DataSpace<3u> const idxClosest
                                = static_cast<pmacc::math::Vector<int, 3u>>(pos + float3_X::create(0.5_X));

                            // the other 7 nearest neighbour indices still have to be found
                            DataSpace<3u> idxShift; // shift to the other nearest neighbour indices
                            DataSpace<3u> idxNext; // nearest neighbour index
                            float3_X weight;
                            for(uint32_t d = 0u; d < 3u; d++)
                            {
                                if(idxClosest[d] == 0) // to avoid border problems
                                    idxShift[d] = 1;
                                else if(pos[d] - static_cast<float_X>(idxClosest[d]) <= 0.0_X)
                                    idxShift[d] = -1;
                                else
                                    idxShift[d] = 1;

                                weight[d] = pmacc::math::abs(static_cast<float_X>(idxClosest[d]) - pos[d]);
                            }

                            // linear interpolation routine
                            float_X dataInterp = 0.0_X;

                            dataInterp
                                += dataBox(idxClosest) * (float3_X::create(1.0_X) - weight).productOfComponents();
                            for(uint32_t d = 0u; d < 3u; d++)
                            {
                                idxNext = idxClosest;
                                float3_X ones = float3_X::create(1.0_X);
                                idxNext[d] = idxClosest[d] + idxShift[d];
                                ones[d] = 0.0_X;
                                dataInterp -= dataBox(idxNext) * (ones - weight).productOfComponents();
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
                                dataInterp += dataBox(idxNext) * (ones - weight).productOfComponents();
                            }
                            dataInterp += dataBox(idxClosest + idxShift) * weight.productOfComponents();

                            return dataInterp;
                        } // linInterpol

                    protected:
                        float_X const timeOriginPIC; // current time at incident plane
                        PMACC_ALIGN(fieldDataBox, typename pmacc::Buffer<float_X, 3u>::DataBoxType);
                        typename pmacc::Buffer<float_X, 1u>::DataBoxType extentOpenPMDdataBox;
                        typename pmacc::Buffer<float_X, 1u>::DataBoxType cellSizeOpenPMDdataBox;
                        typename pmacc::Buffer<float_X, 1u>::DataBoxType offsetOpenPMDdataBox;

                    }; // FromOpenPMDPulseFunctorIncidentE
                } // namespace detail

                template<typename T_Params>
                struct FromOpenPMDPulse
                {
                    //! Get text name of the incident field profile
                    HINLINE static std::string getName()
                    {
                        return "FromOpenPMDPulse";
                    }
                };
            } // namespace profiles

            namespace detail
            {
                /** Get type of incident field E functor for the experimental laser profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentE<profiles::FromOpenPMDPulse<T_Params>>
                {
                    using type = profiles::detail::FromOpenPMDPulseFunctorIncidentE<T_Params>;
                };

                /** Get type of incident field B functor for the experimental laser profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentB<profiles::FromOpenPMDPulse<T_Params>>
                {
                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::FromOpenPMDPulse<T_Params>>::type>;
                };

            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu

#endif
