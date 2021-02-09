/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
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
#include "picongpu/particles/boundary/CallPluginsAndDeleteParticles.hpp"
#include "picongpu/particles/manipulators/manipulators.def"

#include <pmacc/memory/dataTypes/Mask.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/particles/ParticleDescription.hpp>
#include <pmacc/particles/ParticlesBase.hpp>
#include <pmacc/particles/memory/buffers/ParticlesBuffer.hpp>
#include <pmacc/meta/GetKeyFromAlias.hpp>
#include <pmacc/HandleGuardRegion.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/traits/GetCTName.hpp>

#include <boost/mpl/if.hpp>
#include <boost/mpl/contains.hpp>

#include <string>
#include <sstream>
#include <memory>

namespace picongpu
{
    using namespace pmacc;

#if(!BOOST_LANG_CUDA && !BOOST_COMP_HIP)
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
    template<typename T_Name, typename T_Flags, typename T_Attributes>
    class Particles
        : public ParticlesBase<
              ParticleDescription<
                  T_Name,
                  SuperCellSize,
                  T_Attributes,
                  T_Flags,
                  typename bmpl::if_<
                      // check if alias boundaryCondition is defined for the species
                      bmpl::contains<T_Flags, typename GetKeyFromAlias<T_Flags, boundaryCondition<>>::type>,
                      // resolve the alias
                      typename pmacc::traits::Resolve<
                          typename GetKeyFromAlias<T_Flags, boundaryCondition<>>::type>::type,
                      // fallback if the species has not defined the alias boundaryCondition
                      pmacc::HandleGuardRegion<
                          pmacc::particles::policies::ExchangeParticles,
                          particles::boundary::CallPluginsAndDeleteParticles>>::type>,
              MappingDesc,
              DeviceHeap>
        , public ISimulationData
    {
    public:
        using SpeciesParticleDescription = pmacc::ParticleDescription<
            T_Name,
            SuperCellSize,
            T_Attributes,
            T_Flags,
            typename bmpl::if_<
                // check if alias boundaryCondition is defined for the species
                bmpl::contains<T_Flags, typename GetKeyFromAlias<T_Flags, boundaryCondition<>>::type>,
                // resolve the alias
                typename pmacc::traits::Resolve<typename GetKeyFromAlias<T_Flags, boundaryCondition<>>::type>::type,
                // fallback if the species has not defined the alias boundaryCondition
                pmacc::HandleGuardRegion<
                    pmacc::particles::policies::ExchangeParticles,
                    particles::boundary::CallPluginsAndDeleteParticles>>::type>;
        using ParticlesBaseType = ParticlesBase<SpeciesParticleDescription, picongpu::MappingDesc, DeviceHeap>;
        using FrameType = typename ParticlesBaseType::FrameType;
        using FrameTypeBorder = typename ParticlesBaseType::FrameTypeBorder;
        using ParticlesBoxType = typename ParticlesBaseType::ParticlesBoxType;


        Particles(
            const std::shared_ptr<DeviceHeap>& heap,
            picongpu::MappingDesc cellDescription,
            SimulationDataId datasetID);

        void createParticleBuffer();

        void update(uint32_t const currentStep);

        template<typename T_DensityFunctor, typename T_PositionFunctor>
        void initDensityProfile(
            T_DensityFunctor& densityFunctor,
            T_PositionFunctor& positionFunctor,
            const uint32_t currentStep);

        template<
            typename T_SrcName,
            typename T_SrcAttributes,
            typename T_SrcFlags,
            typename T_ManipulateFunctor,
            typename T_SrcFilterFunctor>
        void deviceDeriveFrom(
            Particles<T_SrcName, T_SrcAttributes, T_SrcFlags>& src,
            T_ManipulateFunctor& manipulateFunctor,
            T_SrcFilterFunctor& srcFilterFunctor);

        SimulationDataId getUniqueId() override;

        /* sync device data to host
         *
         * ATTENTION: - in the current implementation only supercell meta data are copied!
         *            - the shared (between all species) mallocMC buffer must be copied once
         *              by the user
         */
        void synchronize() override;

        void syncToDevice() override;

        static pmacc::traits::StringProperty getStringProperties()
        {
            pmacc::traits::StringProperty propList;
            const DataSpace<DIM3> periodic
                = Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();

            for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
            {
                // for each planar direction: left right top bottom back front
                if(FRONT % i == 0)
                {
                    const std::string directionName = ExchangeTypeNames()[i];
                    const DataSpace<DIM3> relDir = Mask::getRelativeDirections<DIM3>(i);

                    const bool isPeriodic = (relDir * periodic) != DataSpace<DIM3>::create(0);

                    std::string boundaryName = "absorbing";
                    if(isPeriodic)
                        boundaryName = "periodic";

                    if(boundaryName == "absorbing")
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

        template<typename T_Pusher>
        void push(uint32_t const currentStep);

    private:
        SimulationDataId m_datasetID;

        /** Get exchange memory size.
         *
         * @param ex exchange index calculated from pmacc::typ::ExchangeType, valid range: [0;27)
         * @return exchange size in bytes
         */
        size_t exchangeMemorySize(uint32_t ex) const;

        FieldE* fieldE;
        FieldB* fieldB;
    };

    namespace traits
    {
        template<typename T_Name, typename T_Attributes, typename T_Flags>
        struct GetDataBoxType<picongpu::Particles<T_Name, T_Attributes, T_Flags>>
        {
            using type = typename picongpu::Particles<T_Name, T_Attributes, T_Flags>::ParticlesBoxType;
        };
    } // namespace traits
} // namespace picongpu

namespace pmacc
{
    namespace traits
    {
        template<typename T_Name, typename T_Flags, typename T_Attributes>
        struct GetCTName<::picongpu::Particles<T_Name, T_Flags, T_Attributes>>
        {
            using type = T_Name;
        };

    } // namespace traits
} // namespace pmacc
