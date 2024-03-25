/* Copyright 2013-2023 Felix Schmitt, Rene Widera, Sergei Bastrakov
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

#if(ENABLE_OPENPMD == 1)

#    pragma once

#    include "picongpu/simulation_defines.hpp"

#    include "picongpu/fields/Fields.hpp"
#    include "picongpu/simulation/control/MovingWindow.hpp"

#    include <pmacc/dataManagement/DataConnector.hpp>
#    include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
#    include <pmacc/memory/buffers/GridBuffer.hpp>
#    include <pmacc/static_assert.hpp>

#    include <algorithm>
#    include <array>
#    include <cstdint>
#    include <functional>
#    include <memory>
#    include <numeric>
#    include <stdexcept>
#    include <string>

#    include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace densityProfiles
    {
        /** Static storage for runtime density for a species
         *
         * Used to propagate runtime values to FromOpenPMDImpl.
         * Same runtime value is used for all FromOpenPMDImpl<> invocations for the same species with no compile-time
         * value set.
         *
         * @tparam T_SpeciesType species type
         */
        template<typename T_SpeciesType>
        struct RuntimeDensityFile
        {
            //! Access the statically stored value
            static std::string& get()
            {
                static auto value = std::string{};
                return value;
            }
        };

        namespace detail
        {
            /** Implementation of loading density from a file
             *
             * @tparam T_ParamClass parameter type
             * @tparam T_SpeciesType species type
             */
            template<typename T_ParamClass, typename T_SpeciesType>
            struct FromOpenPMDImpl : public T_ParamClass
            {
                //! Parameters type
                using ParamClass = T_ParamClass;

                //! Species type
                using SpeciesType = T_SpeciesType;

                /** Initialize the functor on the host side, includes loading the file
                 *
                 * This is MPI collective operation, must be called by all ranks.
                 *
                 * @param currentStep current time iteration
                 */
                HINLINE FromOpenPMDImpl(uint32_t currentStep)
                {
                    auto const& subGrid = Environment<simDim>::get().SubGrid();
                    totalLocalDomainOffset = subGrid.getGlobalDomain().offset + subGrid.getLocalDomain().offset;
                    loadFile();
                }

                /** Calculate the normalized density based on the file contents
                 *
                 * @param totalCellOffset total offset including all slides [in cells]
                 */
                HDINLINE float_X operator()(DataSpace<simDim> const& totalCellOffset)
                {
                    auto const idx = totalCellOffset - totalLocalDomainOffset;
                    return precisionCast<float_X>(deviceDataBox(idx).x());
                }

            private:
                //! Load density from the file to a device buffer
                void loadFile()
                {
                    auto& dc = Environment<>::get().DataConnector();
                    PMACC_CASSERT_MSG(_please_allocate_at_least_one_FieldTmp_in_memory_param, fieldTmpNumSlots > 0);
                    auto fieldTmp = dc.get<FieldTmp>(FieldTmp::getUniqueId(0));
                    auto& fieldBuffer = fieldTmp->getGridBuffer();
                    // Set all values to the default values, the values present in the file will be overwritten
                    fieldBuffer.getHostBuffer().setValue(ParamClass::defaultDensity);
                    auto const guards = fieldBuffer.getGridLayout().getGuard();
                    deviceDataBox = fieldBuffer.getDeviceBuffer().getDataBox().shift(guards);

                    /* Open a series (this does not read the dataset itself).
                     * This is MPI collective and so has to be done by all ranks.
                     */
                    auto& gc = Environment<simDim>::get().GridController();
                    auto const filename = getFilename();
                    log<picLog::PHYSICS>("Loading density for species \"%1%\" from file \"%2%\"")
                        % SpeciesType::FrameType::getName() % filename;
                    auto series
                        = ::openPMD::Series{filename, ::openPMD::Access::READ_ONLY, gc.getCommunicator().getMPIComm()};
                    auto mesh = series.iterations[ParamClass::iteration].meshes[ParamClass::datasetName];
                    ::openPMD::MeshRecordComponent dataset = mesh[::openPMD::RecordComponent::SCALAR];
                    auto const indexConverter = IndexConverter{mesh};
                    auto const datasetExtent = indexConverter.openPMDToXyz(dataset.getExtent());

                    // Offset of the local domain in file coordinates: global coordinates, no guards, no moving window
                    auto const& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
                    bool readFromFile = true;
                    // All indices are in PIConGPU x-y-z coordinates, unless explicitly stated otherwise
                    // Where the fieldBuffer data is starting from (no guards), so inside the local domain
                    auto localDataBoxStart = pmacc::DataSpace<simDim>::create(0);
                    // Start and extend of the file for the local domain
                    auto chunkOffset = ::openPMD::Offset(simDim, 0);
                    auto chunkExtent = ::openPMD::Extent(simDim, 0);
                    for(uint32_t d = 0; d < simDim; ++d)
                    {
                        localDataBoxStart[d] = std::max(ParamClass::offset[d] - totalLocalDomainOffset[d], 0);
                        chunkOffset[d] = std::max(totalLocalDomainOffset[d] - ParamClass::offset[d], 0);
                        // Here we take care, as chunkExtent is unsigned
                        int32_t extent = std::min(
                            static_cast<int32_t>(localDomain.size[d] - localDataBoxStart[d]),
                            static_cast<int32_t>(datasetExtent[d] - chunkOffset[d]));
                        if(extent <= 0)
                            readFromFile = false;
                        else
                            chunkExtent[d] = extent;
                    }

                    using ValueType = FieldTmp::ValueType::type;
                    auto data = std::shared_ptr<ValueType>{nullptr};
                    if(readFromFile)
                    {
                        data = dataset.loadChunk<ValueType>(
                            indexConverter.xyzToOpenPMD(chunkOffset),
                            indexConverter.xyzToOpenPMD(chunkExtent));
                    }
                    // This is MPI collective and so has to be done by all ranks
                    series.flush();

                    if(readFromFile)
                    {
                        auto const* rawData = data.get();
                        auto const numElements = std::accumulate(
                            std::begin(chunkExtent),
                            std::end(chunkExtent),
                            1u,
                            std::multiplies<uint32_t>());
                        auto hostDataBox = fieldBuffer.getHostBuffer().getDataBox().shift(guards + localDataBoxStart);
                        for(uint32_t linearIdx = 0u; linearIdx < numElements; linearIdx++)
                        {
                            auto const idx = indexConverter.linearToXyz(linearIdx, chunkExtent);
                            hostDataBox(idx) = rawData[linearIdx];
                        }
                    }

                    // Copy host data to the device
                    fieldBuffer.hostToDevice();
                    eventSystem::getTransactionEvent().waitForFinished();
                }

                //! Get file name to load density from
                std::string getFilename() const
                {
                    auto const isCompileTimeSet = (ParamClass::filename != "");
                    if(isCompileTimeSet)
                        return ParamClass::filename;
                    else
                        return RuntimeDensityFile<SpeciesType>::get();
                }

                /** Helper class to convert indices between x-y-z coordinates and a system defined by openPMD API mesh
                 *
                 * The latter is defined by a combination of axisLabels and dataOrder of the given mesh object.
                 * Thus, the transformation is bound to the mesh object given in the constructor, not to openPMD API in
                 * general. However, for brevity in this class it is called "openPMD coordinates"
                 */
                class IndexConverter
                {
                public:
                    /** Create an index converter for the given mesh
                     *
                     * @param mesh openPMD API mesh
                     */
                    IndexConverter(::openPMD::Mesh const& mesh)
                    {
                        if(mesh.dataOrder() != ::openPMD::Mesh::DataOrder::C)
                            throw std::runtime_error(
                                "Unsupported dataOrder in FromOpenPMD density dataset, only C is supported");
                        auto axisLabels = std::vector<std::string>{mesh.axisLabels()};
                        // When the attribute is not set, openPMD API currently makes it a vector of single "x"
                        if(axisLabels.size() <= 1)
                            axisLabels = std::vector<std::string>{"x", "y", "z"};
                        std::array<std::string, 3> supportedAxes = {"x", "y", "z"};
                        for(auto d = 0; d < simDim; d++)
                        {
                            auto it = std::find(begin(supportedAxes), begin(supportedAxes) + simDim, axisLabels[d]);
                            if(it != std::end(supportedAxes))
                            {
                                openPMDAxisIndex[d] = std::distance(begin(supportedAxes), it);
                                xyzAxisIndex[openPMDAxisIndex[d]] = d;
                            }
                            else
                                throw std::runtime_error(
                                    "Unsupported axis label " + axisLabels[d] + " in FromOpenPMD density dataset");
                        }
                    }

                    /** Convert a multidimentional index from x-y-z to the openPMD coordinates
                     *
                     * @tparam T_Vector vector type, compatible to std::vector
                     *
                     * @param vector input vector
                     */
                    template<typename T_Vector>
                    T_Vector xyzToOpenPMD(T_Vector const& vector) const
                    {
                        auto result = vector;
                        for(auto d = 0; d < simDim; d++)
                            result[openPMDAxisIndex[d]] = vector[d];
                        return result;
                    }

                    /** Convert a multidimentional index from openPMD to the x-y-z coordinates
                     *
                     * @tparam T_Vector vector type, compatible to std::vector
                     *
                     * @param vector input vector
                     */
                    template<typename T_Vector>
                    T_Vector openPMDToXyz(T_Vector const& vector) const
                    {
                        auto result = vector;
                        for(int32_t d = 0; d < simDim; d++)
                            result[xyzAxisIndex[d]] = vector[d];
                        return result;
                    }

                    /** Convert a linear index in openPMD chunk to a multidimentional x-y-z index.
                     *
                     * @param openPMDLinearIndex linear index inside openPMD chunk
                     * @param xyzChunkExtent multidimentional chunk extent in xyz
                     */
                    pmacc::DataSpace<simDim> linearToXyz(uint32_t openPMDLinearIndex, ::openPMD::Extent xyzChunkExtent)
                        const
                    {
                        // Convert xyz extent to openPMD one
                        auto const openPMDChunkExtent = xyzToOpenPMD(xyzChunkExtent);
                        // This is index in the openPMD coordinate system, the calculation relies on the C data order
                        pmacc::DataSpace<simDim> openPMDIdx;
                        auto tmpIndex = openPMDLinearIndex;
                        for(int32_t d = simDim - 1; d >= 0; d--)
                        {
                            openPMDIdx[d] = tmpIndex % openPMDChunkExtent[d];
                            tmpIndex /= openPMDChunkExtent[d];
                        }
                        // Now we convert it to the xyz coordinates
                        return openPMDToXyz(openPMDIdx);
                    }

                private:
                    // openPMDAxisIndex[0] is openPMD axis index for x, [1] - for y, [2] - for z
                    pmacc::DataSpace<simDim> openPMDAxisIndex;

                    // xyzAxisIndex[0] is x axis index in openPMD, [1] - y, [2] - z
                    pmacc::DataSpace<simDim> xyzAxisIndex;
                };

                /** Device data box with density values
                 *
                 * Density at given totalCellIdx is given by element (totalCellIdx - totalLocalDomainOffset).
                 */
                PMACC_ALIGN(deviceDataBox, FieldTmp::DataBoxType);

                //! Total offset of the local domain in cells
                PMACC_ALIGN(totalLocalDomainOffset, DataSpace<simDim>);
            };
        } // namespace detail

        /** Wrapper to be used in density.param, compatible with other density definitions
         *
         * Hooks internal implementation in detail:: to boost::mpl::apply
         *
         * @tparam T_ParamClass parameter type
         */
        template<typename T_ParamClass>
        struct FromOpenPMDImpl : public T_ParamClass
        {
            template<typename T_SpeciesType>
            struct apply
            {
                using type = detail::FromOpenPMDImpl<T_ParamClass, T_SpeciesType>;
            };
        };

    } // namespace densityProfiles
} // namespace picongpu

#endif
