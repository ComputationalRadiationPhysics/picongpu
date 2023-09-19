/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Marco Garten, Alexander Grund, Sergei Bastrakov
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

#include "picongpu/fields/Fields.def"
#include "picongpu/fields/Fields.hpp"
#include "picongpu/particles/boundary/Description.hpp"
#include "picongpu/particles/boundary/Utility.hpp"
#include "picongpu/particles/manipulators/manipulators.def"

#include <pmacc/HandleGuardRegion.hpp>
#include <pmacc/boundary/Utility.hpp>
#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>
#include <pmacc/meta/GetKeyFromAlias.hpp>
#include <pmacc/particles/ParticleDescription.hpp>
#include <pmacc/particles/ParticlesBase.hpp>
#include <pmacc/particles/memory/buffers/ParticlesBuffer.hpp>
#include <pmacc/particles/policies/DoNothing.hpp>
#include <pmacc/particles/policies/ExchangeParticles.hpp>
#include <pmacc/traits/GetCTName.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/types.hpp>

#include <array>
#include <memory>
#include <sstream>
#include <string>

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
     * @tparam T_Name name of the species [type PMACC_CSTRING]
     * @tparam T_Attributes sequence with attributes [type boost::mp11 list]
     * @tparam T_Flags sequence with flags e.g. solver [type boost::mp11 list]
     */
    template<typename T_Name, typename T_Flags, typename T_Attributes>
    class Particles
        : public ParticlesBase<
              ParticleDescription<
                  T_Name,
                  std::integral_constant<uint32_t, numFrameSlots>,
                  SuperCellSize,
                  T_Attributes,
                  T_Flags,
                  pmacc::mp_if<
                      // check if alias boundaryCondition is defined for the species
                      pmacc::mp_contains<T_Flags, typename GetKeyFromAlias<T_Flags, boundaryCondition<>>::type>,
                      // resolve the alias
                      typename pmacc::traits::Resolve<
                          typename GetKeyFromAlias<T_Flags, boundaryCondition<>>::type>::type,
                      // fallback if the species has not defined the alias boundaryCondition
                      pmacc::HandleGuardRegion<
                          pmacc::particles::policies::ExchangeParticles,
                          pmacc::particles::policies::DoNothing>>>,
              MappingDesc,
              DeviceHeap>
        , public ISimulationData
    {
    public:
        using SpeciesParticleDescription = pmacc::ParticleDescription<
            T_Name,
            std::integral_constant<uint32_t, numFrameSlots>,
            SuperCellSize,
            T_Attributes,
            T_Flags,
            pmacc::mp_if<
                // check if alias boundaryCondition is defined for the species
                pmacc::mp_contains<T_Flags, typename GetKeyFromAlias<T_Flags, boundaryCondition<>>::type>,
                // resolve the alias
                typename pmacc::traits::Resolve<typename GetKeyFromAlias<T_Flags, boundaryCondition<>>::type>::type,
                // fallback if the species has not defined the alias boundaryCondition
                pmacc::HandleGuardRegion<
                    pmacc::particles::policies::ExchangeParticles,
                    pmacc::particles::policies::DoNothing>>>;
        using ParticlesBaseType = ParticlesBase<SpeciesParticleDescription, picongpu::MappingDesc, DeviceHeap>;
        using FrameType = typename ParticlesBaseType::FrameType;
        using FrameTypeBorder = typename ParticlesBaseType::FrameTypeBorder;
        using ParticlesBoxType = typename ParticlesBaseType::ParticlesBoxType;


        Particles(
            const std::shared_ptr<DeviceHeap>& heap,
            picongpu::MappingDesc cellDescription,
            SimulationDataId datasetID);

        void createParticleBuffer();

        //! Push all particles
        void update(uint32_t const currentStep);

        /** Update the supercell storage for particles in the area according to particle attributes
         *
         * @tparam T_MapperFactory factory type to construct a mapper that defines the area to process
         *
         * @param mapperFactory factory instance
         * @param onlyProcessMustShiftSupercells whether to process only supercells with mustShift set to true
         * (optimization to be used with particle pusher) or process all supercells
         */
        template<typename T_MapperFactory>
        inline void shiftBetweenSupercells(T_MapperFactory const& mapperFactory, bool onlyProcessMustShiftSupercells);

        //! Apply all boundary conditions
        void applyBoundary(uint32_t const currentStep);

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

        /** Get boundary descriptions for the species.
         *
         * For both sides along the same axis, both boundaries have the same description.
         * Must not be modified outside of the ParticleBoundaries simulation stage.
         *
         * This method is static as it is used by static getStringProperties().
         */
        static std::array<particles::boundary::Description, simDim>& boundaryDescription()
        {
            static std::array<particles::boundary::Description, simDim> kinds = getDefaultBoundaryDescription();
            return kinds;
        }

        static pmacc::traits::StringProperty getStringProperties()
        {
            pmacc::traits::StringProperty propList;

            for(auto exchange : particles::boundary::getAllAxisAlignedExchanges())
            {
                auto const axis = pmacc::boundary::getAxis(exchange);
                auto const temperature = boundaryDescription()[axis].temperature;

                const std::string directionName = ExchangeTypeNames()[exchange];
                propList[directionName]["param"] = std::string("none");
                switch(boundaryDescription()[axis].kind)
                {
                case particles::boundary::Kind::Periodic:
                    propList[directionName]["name"] = "periodic";
                    break;
                case particles::boundary::Kind::Absorbing:
                    propList[directionName]["name"] = "absorbing";
                    propList[directionName]["param"] = std::string("without field correction");
                    break;
                case particles::boundary::Kind::Reflecting:
                    propList[directionName]["name"] = "reflecting";
                    break;
                case particles::boundary::Kind::Thermal:
                    propList[directionName]["name"] = "reinjecting";
                    propList[directionName]["param"]
                        = std::string("thermal, T=") + std::to_string(temperature) + "keV";
                    break;
                default:
                    propList[directionName]["name"] = "unknown";
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

        //! Get default boundary description for the species matching the communicator topology.
        static std::array<particles::boundary::Description, simDim> getDefaultBoundaryDescription()
        {
            using namespace particles::boundary;
            std::array<Description, simDim> result;
            const DataSpace<DIM3> periodic
                = Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();
            for(uint32_t d = 0; d < simDim; d++)
            {
                result[d].kind = (periodic[d] ? Kind::Periodic : Kind::Absorbing);
                result[d].offset = 0u;
            }
            return result;
        }
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
    namespace lockstep
    {
        //! Specialization to create a lockstep worker configuration out of a particle species.
        template<typename T_Name, typename T_Flags, typename T_Attributes>
        HDINLINE auto makeWorkerCfg(picongpu::Particles<T_Name, T_Flags, T_Attributes> const&)
        {
            return makeWorkerCfg<picongpu::Particles<T_Name, T_Flags, T_Attributes>::FrameType::frameSize>();
        }

        //! Specialization to create a lockstep worker configuration out of a shared pointer to a particle species.
        template<typename T_Name, typename T_Flags, typename T_Attributes>
        HDINLINE auto makeWorkerCfg(std::shared_ptr<picongpu::Particles<T_Name, T_Flags, T_Attributes>> const&)
        {
            return makeWorkerCfg<picongpu::Particles<T_Name, T_Flags, T_Attributes>::FrameType::frameSize>();
        }
    } // namespace lockstep
} // namespace pmacc
