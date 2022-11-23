/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Klaus Steiniger, Felix Schmitt, Benjamin Worpitz
 *                     Finn-Ole Carstens
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

/*

include dis to profile
# path to own libraries
export PIC_LIBS=$HOME/lib

export FFTW3_ROOT=$PIC_LIBS/fftw-3.3.10
export LD_LIBRARY_PATH=$FFTW3_ROOT/lib:$LD_LIBRARY_PATH

*/
#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/plugins/ILightweightPlugin.hpp"
#include "picongpu/plugins/shadowgraphy/ShadowgraphyHelper.hpp"
#include "picongpu/plugins/common/openPMDAttributes.hpp"
#include "picongpu/plugins/common/openPMDDefaultExtension.hpp"
#include "picongpu/plugins/common/openPMDVersion.def"
#include "picongpu/plugins/common/openPMDWriteMeta.hpp"

#include <pmacc/cuSTL/algorithm/host/Foreach.hpp>
#include <pmacc/cuSTL/algorithm/kernel/Foreach.hpp>
#include <pmacc/cuSTL/algorithm/mpi/Gather.hpp>
#include <pmacc/cuSTL/container/DeviceBuffer.hpp>
#include <pmacc/cuSTL/container/HostBuffer.hpp>
#include <pmacc/cuSTL/cursor/tools/slice.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/math/vector/Float.hpp>
#include <pmacc/math/vector/Int.hpp>
#include <pmacc/math/vector/Size_t.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>

#include <openPMD/openPMD.hpp>

#include <iostream>
#include <sstream>
#include <string>

#include <stdio.h>


namespace picongpu
{
    using namespace pmacc;
    namespace po = boost::program_options;

    namespace plugins
    {
        namespace shadowgraphy
        {
            namespace ShadowgraphyHelper
            {
                template<class Field>
                class ConversionFunctor
                {
                public:
                    /* convert field data to higher precision and convert to SI units on GPUs */
                    template<typename T_Acc>
                    DINLINE void operator()(
                        T_Acc const& acc,
                        float3_64& target,
                        const typename Field::ValueType fieldData) const
                    {
                        target = precisionCast<float_64>(fieldData) * float_64((Field::getUnit())[0]);
                    }
                };
            } // end namespace ShadowgraphyHelper

            class Shadowgraphy : public ILightweightPlugin
            {
            private:
                // technical variables for PIConGPU plugins
                std::string pluginName;
                std::string pluginPrefix;
                std::string filenameExtension = openPMD::getDefaultExtension().c_str();

                MappingDesc* cellDescription = nullptr;
                std::string notifyPeriod;

                bool sliceIsOK;
                int plane = 2;
                std::string fileName;
                float_X slicePoint;

                std::unique_ptr<container::DeviceBuffer<float3_64, 2>> dBuffer_SI1;
                std::unique_ptr<container::DeviceBuffer<float3_64, 2>> dBuffer_SI2;

                //std::unique_ptr<::openPMD::Series> openPMDSeries;


                bool isIntegrating;
                int startTime;

                float focuspos;

                int duration;

                bool isMaster = false;

                shadowgraphy::Helper* helper = nullptr;
                pmacc::mpi::MPIReduce reduce;

                bool fourierOutputEnabled = false;
                bool intermediateOutputEnabled = false;

            public:
                Shadowgraphy()
                    : pluginName("Shadowgraphy: calculate the energy density of a laser by integrating"
                                 "the Poynting vectors in a spatially fixed slice over a given time interval")
                    , isIntegrating(false)
                {
                    /* register our plugin during creation */
                    Environment<>::get().PluginConnector().registerPlugin(this);
                    pluginPrefix = "shadowgraphy";
                }

                //! Implementation of base class function.
                std::string pluginGetName() const override
                {
                    return "Shadowgraphy";
                }


                //! Implementation of base class function.
                void pluginRegisterHelp(po::options_description& desc) override
                {
#if(PIC_ENABLE_FFTW3 == 1)
                    desc.add_options()(
                        (this->pluginPrefix + ".period").c_str(),
                        po::value<std::string>(&this->notifyPeriod)->multitoken(),
                        "notify period");
                    desc.add_options()(
                        (this->pluginPrefix + ".fileName").c_str(),
                        po::value<std::string>(&this->fileName)->multitoken(),
                        "file name to store slices in");
                    desc.add_options()(
                        (this->pluginPrefix + ".ext").c_str(),
                        po::value<std::string>(&this->filenameExtension)->multitoken(),
                        "openPMD filename extension");
                    desc.add_options()(
                        (this->pluginPrefix + ".plane").c_str(),
                        po::value<int>(&this->plane)->multitoken(),
                        "specifies the axis which stands on the cutting plane (0,1,2)");
                    desc.add_options()(
                        (this->pluginPrefix + ".slicePoint").c_str(),
                        po::value<float_X>(&this->slicePoint)->multitoken(),
                        "slice point 0.0 <= x <= 1.0");
                    desc.add_options()(
                        (this->pluginPrefix + ".focuspos").c_str(),
                        po::value<float_X>(&this->focuspos)->multitoken(),
                        "focus position relative to slice point in microns");
                    desc.add_options()(
                        (this->pluginPrefix + ".duration").c_str(),
                        po::value<int>(&this->duration)->multitoken(),
                        "nt");
                    desc.add_options()(
                        (this->pluginPrefix + ".fourieroutput").c_str(),
                        po::value<bool>(&fourierOutputEnabled)->zero_tokens(),
                        "optional output: E and B fields in (kx, ky, omega) Fourier space");
                    desc.add_options()(
                        (this->pluginPrefix + ".intermediateoutput").c_str(),
                        po::value<bool>(&intermediateOutputEnabled)->zero_tokens(),
                        "optional output: E and B fields in (x, y, omega) Fourier space");
#else
                    desc.add_options()(
                        (this->pluginPrefix).c_str(),
                        "plugin disabled [compiled without dependency FFTW]");
#endif
                }
            plugins::multi::Option<std::string> extension
                = {"ext", "openPMD filename extension", openPMD::getDefaultExtension().c_str()};

                //! Implementation of base class function.
                void pluginLoad() override
                {
                    /* called when plugin is loaded, command line flags are available here
                     * set notification period for our plugin at the PluginConnector */
                    if(0 != notifyPeriod.size() && float_X(0.0) <= slicePoint && slicePoint <= float_X(1.0))
                    {
                        /* in case the slice point is inside of [0.0,1.0] */
                        sliceIsOK = true;

                        /* The plugin integrates the Poynting vectors over time and must thus be called every tRes-th
                         * time-step of the simulation until the integration is done */
                        int startTime = std::stoi(this->notifyPeriod);
                        int endTime = std::stoi(this->notifyPeriod) + this->duration;

                        std::string internalNotifyPeriod = std::to_string(startTime) + ":" + std::to_string(endTime)
                            + ":" + std::to_string(params::tRes);

                        Environment<>::get().PluginConnector().setNotificationPeriod(this, internalNotifyPeriod);
                        namespace vec = ::pmacc::math;
                        typedef SuperCellSize BlockDim;

                        vec::Size_t<simDim> size = vec::Size_t<simDim>(this->cellDescription->getGridSuperCells())
                                * precisionCast<size_t>(BlockDim::toRT())
                            - precisionCast<size_t>(2 * BlockDim::toRT());
                        this->dBuffer_SI1 = std::make_unique<container::DeviceBuffer<float3_64, simDim - 1>>(
                            size.shrink<simDim - 1>((this->plane + 1) % simDim));
                        this->dBuffer_SI2 = std::make_unique<container::DeviceBuffer<float3_64, simDim - 1>>(
                            size.shrink<simDim - 1>((this->plane + 1) % simDim));
                    }
                    else
                    {
                        /* in case the slice point is outside of [0.0,1.0] */
                        sliceIsOK = false;
                        std::cerr << "In the Shadowgraphy plugin the slice point"
                                  << " (slicePoint=" << slicePoint << ") is outside of [0.0, 1.0]. " << std::endl
                                  << "The request will be ignored. " << std::endl;
                    }
                }

                //! Implementation of base class function.
                void pluginUnload() override
                {
                    /* called when plugin is unloaded, cleanup here */
                }

                /** Implementation of base class function. Sets mapping description.
                 *
                 * @param cellDescription
                 */
                void setMappingDescription(MappingDesc* cellDescription) override
                {
                    this->cellDescription = cellDescription;
                }

                //! Implementation of base class function.
                void notify(uint32_t currentStep) override
                {
                    /* notification callback for simulation step currentStep
                     * called every notifyPeriod steps */
                    if(sliceIsOK)
                    {
                        isMaster = reduce.hasResult(pmacc::mpi::reduceMethods::Reduce());

                        // First time the plugin is called:
                        if(isIntegrating == false)
                        {
                            startTime = currentStep;

                            if(isMaster)
                            {
                                // Get grid size
                                namespace vec = pmacc::math;
                                typedef SuperCellSize BlockDim;
                                DataConnector& dc = Environment<>::get().DataConnector();
                                auto field = dc.get<FieldE>(FieldE::getName(), true)
                                                 ->getGridBuffer()
                                                 .getDeviceBuffer()
                                                 .cartBuffer()
                                                 .view(BlockDim::toRT(), -BlockDim::toRT());

                                pmacc::GridController<simDim>& con
                                    = pmacc::Environment<simDim>::get().GridController();
                                vec::Size_t<simDim> gpuDim = (vec::Size_t<simDim>) con.getGpuNodes();
                                vec::Size_t<simDim> globalGridSize = gpuDim * field.size();

                                helper = new Helper(
                                    currentStep,
                                    this->slicePoint,
                                    this->focuspos * 1e-6,
                                    this->duration,
                                    this->fourierOutputEnabled,
                                    this->intermediateOutputEnabled);
                            }


                            // Create Integrator object %TODO
                            isIntegrating = true;
                        }

                        // convert currentStep (simulation time-step) into localStep for time domain DFT
                        int localStep = (currentStep - startTime) / params::tRes;

                        if(localStep != int(this->duration / params::tRes))
                        {
                            namespace vec = ::pmacc::math;
                            typedef SuperCellSize BlockDim;
                            DataConnector& dc = Environment<>::get().DataConnector();
                            auto field_coreBorderE = dc.get<FieldE>(FieldE::getName(), true)
                                                         ->getGridBuffer()
                                                         .getDeviceBuffer()
                                                         .cartBuffer()
                                                         .view(BlockDim::toRT(), -BlockDim::toRT());

                            storeSlice<FieldE>(
                                field_coreBorderE,
                                this->plane,
                                this->slicePoint,
                                localStep,
                                currentStep);

                            auto field_coreBorderB = dc.get<FieldB>(FieldB::getName(), true)
                                                         ->getGridBuffer()
                                                         .getDeviceBuffer()
                                                         .cartBuffer()
                                                         .view(BlockDim::toRT(), -BlockDim::toRT());

                            storeSlice<FieldB>(
                                field_coreBorderB,
                                this->plane,
                                this->slicePoint,
                                localStep,
                                currentStep);

                            if(isMaster)
                            {
                                helper->calculate_dft(localStep);
                            }
                        }
                        else
                        {
                            if(isMaster)
                            {
                                helper->propagateFields();
                                helper->calculate_shadowgram();

                                std::ostringstream filename;
                                filename << this->fileName << "_" << startTime << ":" << currentStep << ".dat";

                                writeFile(helper->getShadowgram(), filename.str());

                                writeToOpenPMDFile(currentStep);

                                delete(helper);
                            }
                            isIntegrating = false;
                        }
                    }
                }

                /* Stores the field slices from the host on the device. 2 field slices are required to adjust for the
                 * Yee-offset in the plugin.
                 * https://picongpu.readthedocs.io/en/latest/models/AOFDTD.html#maxwell-s-equations-on-the-mesh
                 * The current implementation is based on the (outdated) slice field printer printer and uses custl!
                 * It works, but it's not nice.
                 */
                template<typename Field, typename TField>
                void storeSlice(const TField& field, int nAxis, float slicePoint, int localStep, int currentStep)
                {
                    namespace vec = pmacc::math;

                    pmacc::GridController<simDim>& con = pmacc::Environment<simDim>::get().GridController();
                    vec::Size_t<simDim> gpuDim = (vec::Size_t<simDim>) con.getGpuNodes();
                    vec::Size_t<simDim> globalGridSize = gpuDim * field.size();

                    // FIRST SLICE OF FIELD FOR YEE OFFSET
                    int globalPlane1 = globalGridSize[nAxis] * slicePoint;
                    int localPlane1 = globalPlane1 % field.size()[nAxis];
                    int gpuPlane1 = globalPlane1 / field.size()[nAxis];

                    // SECOND SLICE OF FIELD FOR YEE OFFSET
                    int globalPlane2 = globalGridSize[nAxis] * slicePoint + 1;
                    int localPlane2 = globalPlane2 % field.size()[nAxis];
                    int gpuPlane2 = globalPlane2 / field.size()[nAxis];

                    vec::Int<simDim> nVector(vec::Int<simDim>::create(0));
                    nVector[nAxis] = 1;

                    zone::SphericZone<simDim> gpuGatheringZone1(gpuDim, nVector * gpuPlane1);
                    gpuGatheringZone1.size[nAxis] = 1;

                    zone::SphericZone<simDim> gpuGatheringZone2(gpuDim, nVector * gpuPlane2);
                    gpuGatheringZone2.size[nAxis] = 1;


                    algorithm::mpi::Gather<simDim> gather1(gpuGatheringZone1);

                    algorithm::mpi::Gather<simDim> gather2(gpuGatheringZone2);

                    if(!gather1.participate() && !gather2.participate())
                    {
                        return;
                    }

                    vec::UInt32<3> twistedAxesVec1((nAxis + 1) % 3, (nAxis + 2) % 3, nAxis);

                    // convert data to higher precision and to SI units
                    ShadowgraphyHelper::ConversionFunctor<Field> cf1;
                    algorithm::kernel::RT::Foreach()(
                        dBuffer_SI1->zone(),
                        dBuffer_SI1->origin(),
                        cursor::tools::slice(field.originCustomAxes(twistedAxesVec1)(0, 0, localPlane1)),
                        cf1);


                    vec::UInt32<3> twistedAxesVec2((nAxis + 1) % 3, (nAxis + 2) % 3, nAxis);

                    // convert data to higher precision and to SI units
                    ShadowgraphyHelper::ConversionFunctor<Field> cf2;
                    algorithm::kernel::RT::Foreach()(
                        dBuffer_SI2->zone(),
                        dBuffer_SI2->origin(),
                        cursor::tools::slice(field.originCustomAxes(twistedAxesVec2)(0, 0, localPlane2)),
                        cf2);


                    // copy selected plane from device to host
                    container::HostBuffer<float3_64, simDim - 1> hBuffer1(dBuffer_SI1->size());
                    hBuffer1 = *dBuffer_SI1;

                    // copy selected plane from device to host
                    container::HostBuffer<float3_64, simDim - 1> hBuffer2(dBuffer_SI2->size());
                    hBuffer2 = *dBuffer_SI2;

                    // collect data from all nodes/GPUs
                    vec::Size_t<simDim> globalDomainSize = Environment<simDim>::get().SubGrid().getGlobalDomain().size;
                    vec::Size_t<simDim - 1> globalSliceSize
                        = globalDomainSize.shrink<simDim - 1>((nAxis + 1) % simDim);
                    container::HostBuffer<float3_64, simDim - 1> globalBuffer1(globalSliceSize);
                    container::HostBuffer<float3_64, simDim - 1> globalBuffer2(globalSliceSize);

                    gather1(globalBuffer1, hBuffer1, nAxis);
                    gather2(globalBuffer2, hBuffer2, nAxis);

                    if(!gather1.root() || !gather2.root())
                    {
                        return;
                    }

                    helper->storeField<Field>(localStep, currentStep, &globalBuffer1, &globalBuffer2);
                }

                void writeToOpenPMDFile(uint32_t currentStep)
                {
                    std::stringstream filename;
                    filename << pluginPrefix << "_%T." << filenameExtension;
                    ::openPMD::Series series(filename.str(), ::openPMD::Access::CREATE);
/*
                    ::openPMD::Offset offset = ::openPMD::Offset{0, 0};
                    auto extent = ::openPMD::Extent{  
                        static_cast<unsigned long int>(helper->getSizeX()),  
                        static_cast<unsigned long int>(helper->getSizeY())};

                    auto mesh = series.iterations[currentStep].meshes["shadowgram"];
                    auto shadowgram = mesh[::openPMD::RecordComponent::SCALAR];
*/

                    //typedef pmacc::container::HostBuffer<float_X, DIM2> HBufShdg;

            //this->hBufTotalCalorimeter = std::make_unique<HBufCalorimeter>(this->dBufCalorimeter->size());
            //std::unique_ptr<HBufCalorimeter> hBufTotalCalorimeter;

                    //std::unique_ptr<HBufShdg> hBuf = std::make_unique<HBufShdg>([helper->getSizeX(), helper->getSizeY()]);
                    /*
                    const int size = 10;
                    std::vector<float_64> global_data(size * size);
                    std::iota(global_data.begin(), global_data.end(), 0.);
                    
                    ::openPMD::Extent extent = {size, size};
                    ::openPMD::Offset offset = {0, 0};
                    */
                    
                    ::openPMD::Extent extent = {  
                        static_cast<unsigned long int>(helper->getSizeX()),  
                        static_cast<unsigned long int>(helper->getSizeY())};
                    ::openPMD::Offset offset = {0, 0};
                    

                    ::openPMD::Datatype datatype = ::openPMD::determineDatatype<float_64>();
                    ::openPMD::Dataset dataset{datatype, std::move(extent)};
                   // auto mesh = series.iterations[currentStep].meshes["shadowgram"];
                    //shadowgram.resetDataset(std::move(dataset));

                    ::openPMD::MeshRecordComponent mesh = series.iterations[currentStep].meshes["shadowgram"][::openPMD::MeshRecordComponent::SCALAR];


                    series.flush();
                    mesh.resetDataset(dataset);

                    series.flush();

                    //    std::shared_ptr<float_X>{&(*this->hBufTotalCalorimeter->origin()), [](auto const*) {}},

                    //shadowgram.resetDataset({::openPMD::determineDatatype<float_64>(), extent});
                    mesh.storeChunk(
                        std::shared_ptr<float_64>{&(helper->getShadowgramBuf()->origin()), [](auto const*) {}},
                        std::move(offset),
                        std::move(extent));
                    
                        //std::shared_ptr<float_X>{&(*this->hBufTotalCalorimeter->origin()), [](auto const*) {}},
                        //std::shared_ptr<float_X>{&(helper->getShadowgram()), [](auto const*) {}},
                        //::openPMD::shareRaw(&(helper->getShadowgram()[0][0])), <- compiles
                        //std::shared_ptr<float_64>{&(helper->getShadowgram().front()), [](auto const*) {}},
                        //helper->getShadowgram1D(),
                        //std::make_shared<std::vector<float_64>>(std::move(helper->getShadowgram1D())),
                        //global_data,
                        //::openPMD::shareRaw(&(helper->getShadowgram()->orgin())),
                        //std::shared_ptr<float_X>{(&(helper->getShadowgram()[0][0])), [](auto const*) {}},
                        //std::shared_ptr<float_X>{&(*this->hBufTotalCalorimeter->origin()), [](auto const*) {}},
                        //std::shared_ptr<float_64>{&(*helper.getShadowgramBuf()->origin()), [](auto const*) {}},

                    series.flush();
                    series.iterations[currentStep].close();
                }

                

                void writeFile(std::vector<std::vector<float_64>> values, std::string name)
                {
                    std::ofstream outFile;
                    outFile.open(name.c_str(), std::ofstream::out | std::ostream::trunc);

                    if(!outFile)
                    {
                        std::cerr << "Can't open file [" << name << "] for output, disable plugin output. "
                                  << std::endl;
                        isMaster = false; // no Master anymore -> no process is able to write
                    }
                    else
                    {
                        for(unsigned int i = 0; i < helper->getSizeX(); ++i) // over all x
                        {
                            for(unsigned int j = 0; j < helper->getSizeY(); ++j) // over all y
                            {
                                outFile << values[i][j] << "\t";
                            } // for loop over all y

                            outFile << std::endl;
                        } // for loop over all x

                        outFile.flush();
                        outFile << std::endl; // now all data are written to file

                        if(outFile.fail())
                            std::cerr << "Error on flushing file [" << name << "]. " << std::endl;

                        outFile.close();
                    }
                }
            };

        } // namespace shadowgraphy
    } // namespace plugins
} // namespace picongpu