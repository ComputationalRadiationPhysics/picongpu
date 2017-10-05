/* Copyright 2013-2017 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Marco Garten, Alexander Grund
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

#include <pmacc/types.hpp>

#include "picongpu/fields/Fields.def"
#include "picongpu/fields/Fields.hpp"
#include <pmacc/particles/ParticlesBase.hpp>
#include <pmacc/particles/memory/buffers/ParticlesBuffer.hpp>
#include "picongpu/particles/manipulators/manipulators.def"
#include <pmacc/particles/ParticleDescription.hpp>

#include <pmacc/memory/dataTypes/Mask.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/dataManagement/ISimulationData.hpp>

#include <string>
#include <sstream>
#include <memory>

namespace picongpu
{
using namespace pmacc;

#if( PMACC_CUDA_ENABLED != 1 )
/* dummy because we are not using mallocMC with cupla
 * DeviceHeap is defined in `mallocMC.param`
 */
struct DeviceHeap
{
    using AllocatorHandle = int;

    int getAllocatorHandle()
    {
        return 0;
    }
};
#endif

/** particle species
 *
 * @tparam T_Name name of the species [type boost::mpl::string]
 * @tparam T_Attributes sequence with attributes [type boost::mpl forward sequence]
 * @tparam T_Flags sequence with flags e.g. solver [type boost::mpl forward sequence]
 */
template<
    typename T_Name,
    typename T_Flags,
    typename T_Attributes
>
class Particles : public ParticlesBase<
    ParticleDescription<
        T_Name,
        SuperCellSize,
        T_Attributes,
        T_Flags
    >,
    MappingDesc,
    DeviceHeap
>, public ISimulationData
{
public:

    typedef ParticleDescription<
        T_Name,
        SuperCellSize,
        T_Attributes,
        T_Flags
    > SpeciesParticleDescription;
    typedef ParticlesBase<SpeciesParticleDescription, MappingDesc, DeviceHeap> ParticlesBaseType;
    typedef typename ParticlesBaseType::FrameType FrameType;
    typedef typename ParticlesBaseType::FrameTypeBorder FrameTypeBorder;
    typedef typename ParticlesBaseType::ParticlesBoxType ParticlesBoxType;


    Particles(const std::shared_ptr<DeviceHeap>& heap, MappingDesc cellDescription, SimulationDataId datasetID);

    void createParticleBuffer();

    void update(uint32_t currentStep);

    template<typename T_DensityFunctor, typename T_PositionFunctor>
    void initDensityProfile(T_DensityFunctor& densityFunctor, T_PositionFunctor& positionFunctor, const uint32_t currentStep);

    template<
        typename T_SrcName,
        typename T_SrcAttributes,
        typename T_SrcFlags,
        typename T_ManipulateFunctor
    >
    void deviceDeriveFrom(
        Particles<
            T_SrcName,
            T_SrcAttributes,
            T_SrcFlags
        >& src,
        T_ManipulateFunctor& manipulateFunctor
    );

    template<typename T_Functor>
    void manipulateAllParticles(uint32_t currentStep, T_Functor& functor);

    SimulationDataId getUniqueId();

    /* sync device data to host
     *
     * ATTENTION: - in the current implementation only supercell meta data are copied!
     *            - the shared (between all species) mallocMC buffer must be copied once
     *              by the user
     */
    void synchronize();

    void syncToDevice();

    static pmacc::traits::StringProperty getStringProperties()
    {
        pmacc::traits::StringProperty propList;
        const DataSpace<DIM3> periodic =
            Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();

        for( uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i )
        {
            // for each planar direction: left right top bottom back front
            if( FRONT % i == 0 )
            {
                const std::string directionName = ExchangeTypeNames()[i];
                const DataSpace<DIM3> relDir = Mask::getRelativeDirections<DIM3>(i);

                const bool isPeriodic =
                    (relDir * periodic) != DataSpace<DIM3>::create(0);

                std::string boundaryName = "absorbing";
                if( isPeriodic )
                    boundaryName = "periodic";

                if( boundaryName == "absorbing" )
                {
                    propList[directionName]["param"] = std::string("without field correction");
                }
                else
                {
                    propList[directionName]["param"] = std::string("none");
                }

                propList[directionName]["name"] = boundaryName;
            }
        }
        return propList;
    }

private:
    SimulationDataId m_datasetID;

    FieldE *fieldE;
    FieldB *fieldB;
};

namespace traits
{
    template<
        typename T_Name,
        typename T_Attributes,
        typename T_Flags
    >
    struct GetDataBoxType<
        picongpu::Particles<
            T_Name,
            T_Attributes,
            T_Flags
       >
    >
    {
        typedef typename picongpu::Particles<
            T_Name,
            T_Attributes,
            T_Flags
        >::ParticlesBoxType type;
    };
} //namespace traits
} //namespace picongpu
