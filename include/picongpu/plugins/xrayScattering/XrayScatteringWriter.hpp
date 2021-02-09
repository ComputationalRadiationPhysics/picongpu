/* Copyright 2020-2021 Pawel Ordyna
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
#include <pmacc/assert.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>

#include <openPMD/openPMD.hpp>

#include <vector>
#include <cstdint>

namespace picongpu
{
    namespace plugins
    {
        namespace xrayScattering
        {
            //! Specifies plugin functioning mode. Mirrored or chunked output possible.
            enum class OutputMemoryLayout
            {
                Mirror,
                Distribute
            };


            //! Specifies complex number component.
            enum class Component
            {
                Real,
                Imag
            };


            //! Maps a linear index to a 2D cell position vector.
            HINLINE std::vector<uint64_t> map2d(pmacc::math::Vector<uint64_t, DIM2> const& size, uint64_t pos)
            {
                auto const y(pos % size.y());
                auto const x(pos / size.y());
                return std::vector<uint64_t>{x, y};
            }


            //! Converts a pmacc Vector to an std::Vector.
            template<unsigned DIM, typename T>
            HINLINE std::vector<T> asStandardVector(pmacc::math::Vector<T, DIM> const& vec)
            {
                std::vector<T> res;
                res.reserve(DIM);
                for(unsigned i = 0; i < DIM; ++i)
                {
                    res.push_back(vec[i]);
                }
                return res;
            }


            /** Output writer for the xrayScattering plugin.
             *
             * Handles either a serial, in the mirrored output mode, or a parallel, in
             * the distributed (chunked) mode, data writing. Data is saved in the
             * openPMD standard using the openPMD API.
             * @tparam T_ValueType Type of the values stored in the output.
             */
            template<typename T_ValueType>
            struct XrayScatteringWriter
            {
            private:
                //! A pointer to an openPMD API Series object
                std::unique_ptr<::openPMD::Series> openPMDSeries;
                //! MPI Communicator for the parallel data write
                MPI_Comm mpiCommunicator;
                std::string const fileName, fileExtension, dir;
                std::string const compressionMethod;
                //! Functioning mode
                OutputMemoryLayout outputMemoryLayout;
                //! Output dimensions
                pmacc::math::UInt64<DIM2> const globalExtent;
                //! OpenPMD type specifier for the ValueType
                ::openPMD::Datatype datatype;
                //! Output SI unit
                const float_64 unit;
                //! GridSpacing
                float2_X const gridSpacing;


            public:
                /** Initializes a XrayScatteringWriter object.
                 *
                 * @param fileName Output file name, without the  extensions.
                 * @param fileExtension File extension, specifies the API backend.
                 * @param dir Where to save the output file.
                 * @param outputMemoryLayout  Functioning mode.
                 * @param compressionMethod
                 * @param globalExtent Output dimensions.
                 */
                HINLINE XrayScatteringWriter(
                    std::string const fileName,
                    std::string const fileExtension,
                    std::string const dir,
                    OutputMemoryLayout outputMemoryLayout,
                    std::string const compressionMethod,
                    pmacc::math::UInt64<DIM2> const globalExtent,
                    float2_X const gridSpacing,
                    float_64 const unit,
                    uint32_t const totalSimulationCells)
                    : fileName(fileName)
                    , dir(dir)
                    , fileExtension(fileExtension)
                    , outputMemoryLayout(outputMemoryLayout)
                    , compressionMethod(compressionMethod)
                    , globalExtent(globalExtent)
                    , gridSpacing(gridSpacing)
                    , unit(unit)
                {
                    if(outputMemoryLayout == OutputMemoryLayout::Distribute)
                    {
                        // Set the MPI communicator.
                        GridController<simDim>& gc = Environment<simDim>::get().GridController();
                        __getTransactionEvent().waitForFinished();
                        mpiCommunicator = MPI_COMM_NULL;
                        MPI_CHECK(MPI_Comm_dup(gc.getCommunicator().getMPIComm(), &mpiCommunicator));
                    }

                    datatype = ::openPMD::determineDatatype<T_ValueType>();
                    // Create the output file.
                    openSeries(::openPMD::Access::CREATE);
                    openPMDSeries->setMeshesPath("scatteringData");
                    openPMDSeries->setAttribute("totalSimulationCells", totalSimulationCells);
                    closeSeries();
                }

                virtual ~XrayScatteringWriter()
                {
                    if(outputMemoryLayout == OutputMemoryLayout::Distribute)
                    {
                        if(mpiCommunicator != MPI_COMM_NULL)
                        {
                            // avoid deadlock between not finished pmacc tasks and mpi
                            // blocking collectives
                            __getTransactionEvent().waitForFinished();
                            MPI_CHECK_NO_EXCEPT(MPI_Comm_free(&(mpiCommunicator)));
                        }
                    }
                }

            private:
                HINLINE bool isADIOS1() const
                {
#if openPMD_HAVE_ADIOS1 && !openPMD_HAVE_ADIOS2
                    return this->fileExtension == "bp";
#else
                    return false;
#endif
                }

                /** Opens an openPMD Series in a given access mode.
                 *
                 * @param at OpenPMD API access type.
                 */
                HINLINE void openSeries(::openPMD::Access at)
                {
                    if(!openPMDSeries)
                    {
                        std::string fullName = dir + '/' + fileName + "." + fileExtension;
                        log<picLog::INPUT_OUTPUT>("XrayScatteringWriter: Opening file: %1%") % fullName;

                        if(outputMemoryLayout == OutputMemoryLayout::Distribute)
                        {
                            // Open a series for a parallel write.
                            openPMDSeries = std::make_unique<::openPMD::Series>(fullName, at, mpiCommunicator);
                        }
                        else
                        {
                            // Open a series for a serial write.
                            openPMDSeries = std::make_unique<::openPMD::Series>(fullName, at);
                        }

                        log<picLog::INPUT_OUTPUT>("XrayScatteringWriter: Successfully opened file: %1%") % fullName;
                    }
                    else
                    {
                        throw std::runtime_error("XrayScatteringWriter: Tried opening a Series while old "
                                                 "Series was still active.");
                    }
                }

                HINLINE void closeSeries()
                {
                    if(openPMDSeries)
                    {
                        log<picLog::INPUT_OUTPUT>("XrayScatteringWriter: Closing "
                                                  "file: %1%")
                            % fileName;
                        openPMDSeries.reset();
                        if(outputMemoryLayout == OutputMemoryLayout::Distribute)
                        {
                            MPI_Barrier(mpiCommunicator);
                        }
                        log<picLog::INPUT_OUTPUT>("XrayScatteringWriter: successfully closed file: %1%") % fileName;
                    }
                    else
                    {
                        throw std::runtime_error("XrayScatteringWriter: Tried closing a Series that was not"
                                                 " active.");
                    }
                }


                /** Prepare an openPMD mesh for the amplitude.
                 * @param currentStep
                 */
                HINLINE ::openPMD::Mesh prepareMesh(uint32_t const currentStep)
                {
                    ::openPMD::Iteration iteration = openPMDSeries->iterations[currentStep];
                    ::openPMD::Mesh mesh = iteration.meshes["amplitude"];
                    mesh.setGridSpacing(asStandardVector<DIM2>(gridSpacing));
                    // 1/angstrom to 1/meter conversion
                    mesh.setGridUnitSI(1e10);
                    mesh.setAxisLabels(std::vector<std::string>{"q_x", "q_y"});
                    return mesh;
                }


                /**
                 * @param currentStep
                 * @param component Component to write, either real or imaginary
                 */
                HINLINE ::openPMD::MeshRecordComponent prepareMRC(Component component, ::openPMD::Mesh& mesh)
                {
                    const std::string name_lookup_tpl[] = {"x", "y"};
                    ::openPMD::MeshRecordComponent mrc = mesh[name_lookup_tpl[static_cast<int>(component)]];

                    std::vector<uint64_t> shape = asStandardVector<DIM2>(globalExtent);
                    ::openPMD::Dataset dataset{datatype, std::move(shape)};

                    if(isADIOS1())
                    {
                        dataset.transform = compressionMethod;
                    }
                    else
                    {
                        dataset.compression = compressionMethod;
                    }
                    mrc.resetDataset(std::move(dataset));
                    mrc.setUnitSI(unit);
                    return mrc;
                }

            public:
                /** Write complex numbers to the whole output array.
                 *
                 * @param currentStep Current simulation step.
                 * @param realVec Vector containing the real parts of the complex
                 *      numbers.
                 * @param imagVec Vector containing the imaginary parts of the
                 *      complex numbers.
                 */
                HINLINE void operator()(
                    uint32_t const currentStep,
                    std::vector<T_ValueType>& realVec,
                    std::vector<T_ValueType>& imagVec)
                {
                    openSeries(::openPMD::Access::READ_WRITE);

                    ::openPMD::Mesh mesh = prepareMesh(currentStep);
                    ::openPMD::MeshRecordComponent mrc_real = prepareMRC(Component::Real, mesh);
                    ::openPMD::MeshRecordComponent mrc_imag = prepareMRC(Component::Imag, mesh);


                    mrc_real.storeChunk<T_ValueType>(
                        ::openPMD::shareRaw(&realVec[0]),
                        ::openPMD::Offset(DIM2, 0u),
                        asStandardVector<DIM2>(globalExtent));
                    mrc_imag.storeChunk<T_ValueType>(
                        ::openPMD::shareRaw(&imagVec[0]),
                        ::openPMD::Offset(DIM2, 0u),
                        asStandardVector<DIM2>(globalExtent));
                    openPMDSeries->flush();

                    // Avoid deadlock between not finished pmacc tasks and mpi calls in
                    // openPMD.
                    __getTransactionEvent().waitForFinished();
                    // Close openPMD Series, most likely the actual write point.
                    closeSeries();
                }


                /** Write complex numbers to a part of the output array.
                 *
                 * @param currentStep Current simulation step.
                 * @param extent1D The length of the contiguous part of the output
                 *      that is the write destination (1D access).
                 * @param offset1D The linear (1D access) offset to the first datum
                 *      in the write destination.
                 * @param realVec Vector containing the real parts of the complex
                 *      numbers.
                 * @param imagVec Vector containing the imaginary parts of the
                 *      complex numbers.
                 */
                HINLINE void operator()(
                    uint32_t const currentStep,
                    uint64_t extent1D,
                    uint64_t offset1D,
                    std::vector<T_ValueType>& realVec,
                    std::vector<T_ValueType>& imagVec)
                {
                    openSeries(::openPMD::Access::READ_WRITE);

                    // Get openPMD mesh record components for the real and imaginary
                    // parts.
                    ::openPMD::Mesh mesh = prepareMesh(currentStep);
                    ::openPMD::MeshRecordComponent mrc_real = prepareMRC(Component::Real, mesh);
                    ::openPMD::MeshRecordComponent mrc_imag = prepareMRC(Component::Imag, mesh);

                    // Register chunks to write:
                    // Since the extent1D and offset1D are indices used in a linear
                    // access to the array (along last axis, C-order), they don't always
                    // describe a rectangle in the 2D output space. For that reason it
                    // is in general not possible to specify the write extend with a
                    // 2D vector as it is required by the API. Here the output
                    // destination is split into 3 parts. Two, not full, rows one at
                    // the begining and one ad the end of the chunk and a rectangular
                    // chunk in between.
                    //

                    std::vector<uint64_t> offset(2);
                    std::vector<uint64_t> extent(2);
                    // First line.
                    // Map the beginning of the output chunk.
                    offset = map2d(globalExtent, offset1D);

                    // The first line has not always the maximum possible length.
                    uint64_t firstLineLength = globalExtent[1] - offset[1];
                    // Set the extent vector.
                    extent = std::vector<uint64_t>{1, firstLineLength};
                    // Register chunks for imag and real components.
                    mrc_real.storeChunk<T_ValueType>(::openPMD::shareRaw(&realVec[0]), offset, extent);
                    mrc_imag.storeChunk<T_ValueType>(::openPMD::shareRaw(&imagVec[0]), offset, extent);

                    // Middle chunk.
                    // These lines have the full length.
                    uint64_t numFullLines = (extent1D - firstLineLength) / globalExtent[1];
                    extent[0] = numFullLines;
                    extent[1] = globalExtent[1];
                    // Offset to the middle chunk.
                    uint64_t localOffset = firstLineLength;
                    offset = map2d(globalExtent, offset1D + localOffset);
                    // Register the middle chunk.
                    mrc_real.storeChunk<T_ValueType>(::openPMD::shareRaw(&realVec[localOffset]), offset, extent);
                    mrc_imag.storeChunk<T_ValueType>(::openPMD::shareRaw(&imagVec[localOffset]), offset, extent);

                    // Last line:
                    // Find out the length of the last line in the 1D chunk.
                    uint64_t lastLineLength((extent1D - firstLineLength - numFullLines * globalExtent[1]));
                    if(lastLineLength != 0)
                    {
                        localOffset = firstLineLength + numFullLines * globalExtent[1];
                        offset = map2d(globalExtent, offset1D + localOffset);
                        extent[0] = 1;
                        extent[1] = lastLineLength;
                        mrc_real.storeChunk<T_ValueType>(::openPMD::shareRaw(&realVec[localOffset]), offset, extent);
                        mrc_imag.storeChunk<T_ValueType>(::openPMD::shareRaw(&imagVec[localOffset]), offset, extent);
                    }
                    openPMDSeries->flush();
                    // Avoid deadlock between not finished pmacc tasks and mpi calls in
                    // openPMD.
                    __getTransactionEvent().waitForFinished();
                    // Close the openPMD Series, most likely the actual write point.
                    closeSeries();
                }
            };
        } // namespace xrayScattering
    } // namespace plugins
} // namespace picongpu
