/* Copyright 2021 Pawel Ordyna
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

#include "picongpu/plugins/common/asStandardVector.hpp"
#include "picongpu/plugins/common/openPMDVersion.def"
#include "picongpu/plugins/common/openPMDWriteMeta.hpp"

#include <pmacc/assert.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/static_assert.hpp>

#include <cstdint>
#include <vector>

#include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace plugins
    {
        namespace photonDetector
        {
            /** Output writer for the photonDetector plugin.
             *
             * Data is written in serial mode.
             * @tparam T_ValueType Type of the values stored in the output.
             */
            template<typename T_ValueType>
            struct PhotonDetectorWriter
            {
            private:
                //! a pointer to an openPMD API Series object
                std::unique_ptr<::openPMD::Series> openPMDSeries_m;
                // strings defining the path to the series
                std::string const fileName_m, fileExtension_m, dir_m, fileInfix_m;
                //! backend specific configuration
                std::string const jsonConfig_m;
                //! detector size
                pmacc::math::UInt64<DIM2> const globalExtent_m;
                //! OpenPMD type specifier for the ValueType
                ::openPMD::Datatype datatype_m;
                //! output SI unit
                const float_64 unit_m;
                //! detector cell size
                float2_X const gridSpacing_m;
                //! unit dimension of the stored values
                std::map<::openPMD::UnitDimension, double> const unitMap_m;
                std::string const accumulationPolicyName_m;
                std::string const detectorPlacement_m;
                float_64 const detectorDistance_m;

                /** openPMD series iteration encoding
                 * openPMD api can store iterations:
                 *  - in separate files (file based encoding)
                 *  - in the same file (group based encoding)
                 *  - send iterations via streaming api (value based encoding)
                 */
                ::openPMD::IterationEncoding iterationEncoding_m;
                bool const isMaster_m;
                std::string meshName_m;

            public:
                /** Initializes a PhotonDetectorWriter object.
                 *
                 * @param fileName output file name, without the  extensions.
                 * @param fileInfix openPMD filename infix (use to pick iteration encoding)
                 * @param fileExtension file extension, specifies the API backend.
                 * @param dir directory in simOutput where the output files are stored
                 * @param meshName mesh name for the stored data
                 * @param globalExtent detector size
                 * @param gridSpacing detector cell size
                 * @param unit output SI unit
                 * @param unitDimension unit dimension of the stored values
                 */
                HINLINE PhotonDetectorWriter(
                    bool const& isMaster,
                    std::string const& fileName,
                    std::string const& fileInfix,
                    std::string const& fileExtension,
                    std::string const& dir,
                    std::string const& meshName,
                    pmacc::math::UInt64<DIM2> const& globalExtent,
                    float2_X const& gridSpacing,
                    float_64 const& unit,
                    std::vector<float_64> const& unitDimension,
                    std::string const& accumulationPolicyName,
                    std::string const& detectorPlacement,
                    float_64 const& detectorDistance)
                    : isMaster_m(isMaster)
                    , fileName_m(fileName)
                    , fileInfix_m(fileInfix)
                    , dir_m(dir)
                    , meshName_m(meshName)
                    , fileExtension_m(fileExtension)
                    , globalExtent_m(globalExtent)
                    , gridSpacing_m(gridSpacing)
                    , unit_m(unit)
                    , unitMap_m(openPMD::convertToUnitDimension(unitDimension))
                    , accumulationPolicyName_m(accumulationPolicyName)
                    , detectorPlacement_m(detectorPlacement)
                    , detectorDistance_m(detectorDistance)
                {
                    datatype_m = ::openPMD::determineDatatype<T_ValueType>();

                    // Create output directory and set permissions
                    pmacc::Filesystem<simDim>& fs = Environment<simDim>::get().Filesystem();
                    if(isMaster_m)
                    {
                        fs.createDirectory(dir_m);
                        fs.setDirectoryPermissions(dir_m);
                    }
                    // Determine iteration encoding
                    try
                    {
                        openSeries(::openPMD::Access::READ_ONLY);
                    }
                    catch(...)
                    {
                        openSeries(::openPMD::Access::CREATE);
                    }
                    iterationEncoding_m = openPMDSeries_m->iterationEncoding();
                    switch(iterationEncoding_m)
                    {
                    case ::openPMD::IterationEncoding::fileBased:
                        break;
                    case ::openPMD::IterationEncoding::groupBased:
                        openPMDSeries_m->flush();
                        break;
                    default:
                        throw std::runtime_error(
                            "PhotonDetectorWriter: Internal error. Unrecognized iteration encoding.");
                    }
                    closeSeries();
                }

            private:
                /** Opens the openPMD series.
                 *
                 * @param at openPMD API access type.
                 */
                HINLINE void openSeries(::openPMD::Access at)
                {
                    if(!openPMDSeries_m)
                    {
                        std::string fullName = dir_m + '/' + fileName_m + fileInfix_m + "." + fileExtension_m;
                        log<picLog::INPUT_OUTPUT>("PhotonDetectorWriter: Opening file: %1%") % fullName;

                        // Open a series for a serial write.
                        openPMDSeries_m = std::make_unique<::openPMD::Series>(fullName, at);

                        log<picLog::INPUT_OUTPUT>("PhotonDetectorWriter: Successfully opened file: %1%") % fullName;
                    }
                    else
                    {
                        throw std::runtime_error("PhotonDetectorWriter: Tried opening a Series while old "
                                                 "Series was still active.");
                    }
                }

                //! Closes the openPMD series
                HINLINE void closeSeries()
                {
                    if(openPMDSeries_m)
                    {
                        log<picLog::INPUT_OUTPUT>("PhotonDetectorWriter: Closing "
                                                  "openPMDSeries: %1%")
                            % (fileName_m + fileInfix_m + "." + fileExtension_m);
                        openPMDSeries_m.reset();
                        log<picLog::INPUT_OUTPUT>("PhotonDetectorWriter: successfully closed openPMD series: %1%")
                            % (fileName_m + fileInfix_m + "." + fileExtension_m);
                    }
                    else
                    {
                        throw std::runtime_error("PhotonDetectorWriter: Tried closing a Series that was not"
                                                 " active.");
                    }
                }

                /** Prepare an openPMD iteration for storage
                 *
                 * @ param currentStep
                 */
                HINLINE ::openPMD::Iteration prepareIteration(uint32_t const currentStep)
                {
                    ::openPMD::Iteration iteration = openPMDSeries_m->WRITE_ITERATIONS[currentStep];
                    iteration.setDt(DELTA_T);
                    iteration.setTimeUnitSI(UNIT_TIME);
                    return iteration;
                }

                /** Prepare an openPMD mesh for storage
                 *
                 * @param iteration
                 */
                HINLINE ::openPMD::Mesh prepareMesh(::openPMD::Iteration& iteration)
                {
                    ::openPMD::Mesh mesh = iteration.meshes[meshName_m];

                    mesh.setGridSpacing(openPMD::asStandardVector(gridSpacing_m));
                    mesh.setAxisLabels(std::vector<std::string>{"y", "x"});
                    mesh.setGridGlobalOffset(std::vector<float_64>{0.0, 0.0});
                    mesh.setGeometry(::openPMD::Mesh::Geometry::cartesian);
                    mesh.setDataOrder(::openPMD::Mesh::DataOrder::C);
                    mesh.setTimeOffset(0.0_X);
                    mesh.setGridUnitSI(UNIT_LENGTH);
                    mesh.setUnitDimension(unitMap_m);
                    mesh.setAttribute("accumulationPolicy", accumulationPolicyName_m);
                    mesh.setAttribute("detectorPlacement", detectorPlacement_m);
                    mesh.setAttribute("detectorDistanceInMeters", detectorDistance_m);
                    return mesh;
                }

                /** Prepare an openPMD mesh record component for storage
                 *
                 * @param mesh
                 */
                HINLINE ::openPMD::MeshRecordComponent prepareMRC(::openPMD::Mesh& mesh)
                {
                    ::openPMD::MeshRecordComponent mrc = mesh[::openPMD::MeshRecordComponent::SCALAR];

                    std::vector<uint64_t> shape = openPMD::asStandardVector(globalExtent_m);
                    ::openPMD::Dataset dataset{datatype_m, std::move(shape)};
                    mrc.resetDataset(std::move(dataset));
                    mrc.setPosition(std::vector<float_X>{0.5_X, 0.5_X});
                    mrc.setUnitSI(unit_m);
                    return mrc;
                }

            public:
                /** Write data to the  output array
                 *
                 * @param currentStep current simulation step
                 * @param pointer base pointer to the data that should be written
                 */
                HINLINE void operator()(uint32_t const currentStep, T_ValueType* pointer)
                {
                    // read & write access type is not possible when the iterations are stored in separate files
                    switch(iterationEncoding_m)
                    {
                    case ::openPMD::IterationEncoding::fileBased:
                        openSeries(::openPMD::Access::CREATE);
                        break;
                    case ::openPMD::IterationEncoding::groupBased:
                        openSeries(::openPMD::Access::READ_WRITE);
                        break;
                    // there is also variableBased encoding on the API's dev branch, not sure what it is and when
                    // would it appear in a release
                    default:
                        throw std::runtime_error(
                            "PhotonDetectorWriter: Internal error. Unrecognized iteration encoding.");
                    }
                    ::openPMD::Iteration iteration = prepareIteration(currentStep);
                    ::openPMD::Mesh mesh = prepareMesh(iteration);
                    ::openPMD::MeshRecordComponent mrc = prepareMRC(mesh);

                    mrc.storeChunk<T_ValueType>(
                        ::openPMD::shareRaw(pointer),
                        ::openPMD::Offset(DIM2, 0u),
                        openPMD::asStandardVector(globalExtent_m));

                    // Write simulation meta data
                    openPMD::WriteMeta()(*openPMDSeries_m, currentStep);
                    openPMDSeries_m->flush();

                    // Avoid deadlock between not finished pmacc tasks and mpi calls in
                    // openPMD.
                    __getTransactionEvent().waitForFinished();
                    // Close openPMD Series, most likely the actual write point.
                    closeSeries();
                }
            };
        } // namespace photonDetector
    } // namespace plugins
} // namespace picongpu
