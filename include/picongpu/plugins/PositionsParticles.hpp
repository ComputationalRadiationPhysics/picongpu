/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Benjamin Worpitz, Richard Pausch
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
#include "picongpu/algorithms/Gamma.hpp"
#include "picongpu/plugins/ILightweightPlugin.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"

#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/traits/HasIdentifiers.hpp>
#include <pmacc/traits/HasFlag.hpp>

#include <boost/mpl/and.hpp>

#include <string>
#include <iostream>


namespace picongpu
{
    using namespace pmacc;

    namespace po = boost::program_options;

    template<class FloatPos>
    struct SglParticle
    {
        FloatPos position;
        float3_X momentum;
        float_X mass;
        float_X weighting;
        float_X charge;
        float_X gamma;

        SglParticle()
            : position(FloatPos::create(0.0))
            , momentum(float3_X::create(0.0))
            , mass(0.0)
            , weighting(0.0)
            , charge(0.0)
            , gamma(0.0)
        {
        }

        DataSpace<simDim> globalCellOffset;

        //! todo

        floatD_64 getGlobalCell() const
        {
            floatD_64 doubleGlobalCellOffset;
            for(uint32_t i = 0; i < simDim; ++i)
                doubleGlobalCellOffset[i] = float_64(globalCellOffset[i]);

            return floatD_64(doubleGlobalCellOffset + precisionCast<float_64>(position));
        }

        template<typename T>
        friend std::ostream& operator<<(std::ostream& out, const SglParticle<T>& v)
        {
            floatD_64 pos;
            for(uint32_t i = 0; i < simDim; ++i)
                pos[i] = (v.getGlobalCell()[i] * cellSize[i] * UNIT_LENGTH);

            const float3_64 mom(
                precisionCast<float_64>(v.momentum.x()) * UNIT_MASS * UNIT_SPEED,
                precisionCast<float_64>(v.momentum.y()) * UNIT_MASS * UNIT_SPEED,
                precisionCast<float_64>(v.momentum.z()) * UNIT_MASS * UNIT_SPEED);

            const float_64 mass = precisionCast<float_64>(v.mass) * UNIT_MASS;
            const float_64 charge = precisionCast<float_64>(v.charge) * UNIT_CHARGE;

            using dbl = std::numeric_limits<float_64>;
            out.precision(dbl::digits10);

            out << std::scientific << pos << " " << mom << " " << mass << " " << precisionCast<float_64>(v.weighting)
                << " " << charge << " " << precisionCast<float_64>(v.gamma);
            return out;
        }
    };

    /** write the position of a single particle to a file
     * \warning this plugin MUST NOT be used with more than one (global!)
     * particle and is created for one-particle-test-purposes only
     */
    struct KernelPositionsParticles
    {
        template<typename ParBox, typename FloatPos, typename Mapping, typename T_Acc>
        DINLINE void operator()(T_Acc const& acc, ParBox pb, SglParticle<FloatPos>* gParticle, Mapping mapper) const
        {
            using FramePtr = typename ParBox::FramePtr;
            PMACC_SMEM(acc, frame, FramePtr);


            using SuperCellSize = typename Mapping::SuperCellSize;

            const DataSpace<simDim> threadIndex(cupla::threadIdx(acc));
            const int linearThreadIdx = DataSpaceOperations<simDim>::template map<SuperCellSize>(threadIndex);
            const DataSpace<simDim> superCellIdx(mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(acc))));

            if(linearThreadIdx == 0)
            {
                frame = pb.getLastFrame(superCellIdx);
            }

            cupla::__syncthreads(acc);
            if(!frame.isValid())
                return; // end kernel if we have no frames

            /* BUGFIX to issue #538
             * volatile prohibits that the compiler creates wrong code*/
            volatile bool isParticle = frame[linearThreadIdx][multiMask_];

            while(frame.isValid())
            {
                if(isParticle)
                {
                    auto particle = frame[linearThreadIdx];
                    gParticle->position = particle[position_];
                    gParticle->momentum = particle[momentum_];
                    gParticle->weighting = particle[weighting_];
                    gParticle->mass = attribute::getMass(gParticle->weighting, particle);
                    gParticle->charge = attribute::getCharge(gParticle->weighting, particle);
                    gParticle->gamma = Gamma<>()(gParticle->momentum, gParticle->mass);

                    // storage number in the actual frame
                    const lcellId_t frameCellNr = particle[localCellIdx_];

                    // offset in the actual superCell = cell offset in the supercell
                    const DataSpace<simDim> frameCellOffset(
                        DataSpaceOperations<simDim>::template map<MappingDesc::SuperCellSize>(frameCellNr));


                    gParticle->globalCellOffset
                        = (superCellIdx - mapper.getGuardingSuperCells()) * MappingDesc::SuperCellSize::toRT()
                        + frameCellOffset;
                }
                cupla::__syncthreads(acc);
                if(linearThreadIdx == 0)
                {
                    frame = pb.getPreviousFrame(frame);
                }
                isParticle = true;
                cupla::__syncthreads(acc);
            }
        }
    };

    template<class ParticlesType>
    class PositionsParticles : public ILightweightPlugin
    {
    private:
        typedef MappingDesc::SuperCellSize SuperCellSize;
        typedef floatD_X FloatPos;

        GridBuffer<SglParticle<FloatPos>, DIM1>* gParticle;

        MappingDesc* cellDescription;
        std::string notifyPeriod;

        std::string pluginName;
        std::string pluginPrefix;

    public:
        PositionsParticles()
            : pluginName("PositionsParticles: write position of one particle of a species to std::cout")
            , pluginPrefix(ParticlesType::FrameType::getName() + std::string("_position"))
            , gParticle(nullptr)
            , cellDescription(nullptr)
        {
            Environment<>::get().PluginConnector().registerPlugin(this);
        }

        virtual ~PositionsParticles()
        {
        }

        void notify(uint32_t currentStep)
        {
            const int rank = Environment<simDim>::get().GridController().getGlobalRank();
            const SglParticle<FloatPos> positionParticle = getPositionsParticles<CORE + BORDER>(currentStep);

            /*FORMAT OUTPUT*/
            if(positionParticle.mass != float_X(0.0))
                std::cout << "[ANALYSIS] [" << rank << "] [COUNTER] [" << pluginPrefix << "] [" << currentStep << "] "
                          << std::setprecision(16) << float_64(currentStep) * SI::DELTA_T_SI << " " << positionParticle
                          << "\n"; // no flush
        }

        void pluginRegisterHelp(po::options_description& desc)
        {
            desc.add_options()(
                (pluginPrefix + ".period").c_str(),
                po::value<std::string>(&notifyPeriod),
                "enable plugin [for each n-th step]");
        }

        std::string pluginGetName() const
        {
            return pluginName;
        }

        void setMappingDescription(MappingDesc* cellDescription)
        {
            this->cellDescription = cellDescription;
        }

    private:
        void pluginLoad()
        {
            if(!notifyPeriod.empty())
            {
                // create one float3_X on gpu und host
                gParticle = new GridBuffer<SglParticle<FloatPos>, DIM1>(DataSpace<DIM1>(1));

                Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
            }
        }

        void pluginUnload()
        {
            __delete(gParticle);
        }

        template<uint32_t AREA>
        SglParticle<FloatPos> getPositionsParticles(uint32_t currentStep)
        {
            typedef typename MappingDesc::SuperCellSize SuperCellSize;
            SglParticle<FloatPos> positionParticleTmp;

            DataConnector& dc = Environment<>::get().DataConnector();
            auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName(), true);

            gParticle->getDeviceBuffer().setValue(positionParticleTmp);
            auto block = SuperCellSize::toRT();

            AreaMapping<AREA, MappingDesc> mapper(*cellDescription);
            PMACC_KERNEL(KernelPositionsParticles{})
            (mapper.getGridDim(),
             block)(particles->getDeviceParticlesBox(), gParticle->getDeviceBuffer().getBasePointer(), mapper);

            dc.releaseData(ParticlesType::FrameType::getName());
            gParticle->deviceToHost();

            DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
            const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);

            DataSpace<simDim> gpuPhyCellOffset(Environment<simDim>::get().SubGrid().getLocalDomain().offset);
            gpuPhyCellOffset.y() += (localSize.y() * numSlides);

            gParticle->getHostBuffer().getDataBox()[0].globalCellOffset += gpuPhyCellOffset;


            return gParticle->getHostBuffer().getDataBox()[0];
        }
    };

    namespace particles
    {
        namespace traits
        {
            template<typename T_Species, typename T_UnspecifiedSpecies>
            struct SpeciesEligibleForSolver<T_Species, PositionsParticles<T_UnspecifiedSpecies>>
            {
                using FrameType = typename T_Species::FrameType;

                using RequiredIdentifiers = MakeSeq_t<weighting, momentum, position<>>;

                using SpeciesHasIdentifiers =
                    typename pmacc::traits::HasIdentifiers<FrameType, RequiredIdentifiers>::type;

                using SpeciesHasMass = typename pmacc::traits::HasFlag<FrameType, massRatio<>>::type;
                using SpeciesHasCharge = typename pmacc::traits::HasFlag<FrameType, chargeRatio<>>::type;

                using type = typename bmpl::and_<SpeciesHasIdentifiers, SpeciesHasMass, SpeciesHasCharge>;
            };
        } // namespace traits
    } // namespace particles
} // namespace picongpu
