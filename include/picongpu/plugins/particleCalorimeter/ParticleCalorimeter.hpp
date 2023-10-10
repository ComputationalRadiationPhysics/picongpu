/* Copyright 2016-2023 Heiko Burau, Rene Widera, Sergei Bastrakov,
 * Richard Pausch
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

#include "ParticleCalorimeter.kernel"
#include "ParticleCalorimeterFunctors.hpp"
#include "picongpu/particles/boundary/Utility.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/common/openPMDAttributes.hpp"
#include "picongpu/plugins/common/openPMDDefaultExtension.hpp"
#include "picongpu/plugins/common/openPMDWriteMeta.hpp"
#include "picongpu/plugins/misc/misc.hpp"
#include "picongpu/plugins/multi/multi.hpp"

#include <pmacc/algorithms/math.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/particles/policies/ExchangeParticles.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/HasIdentifiers.hpp>

#include <boost/filesystem.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <openPMD/openPMD.hpp>


namespace picongpu
{
    using namespace pmacc;

    namespace po = boost::program_options;


    /** Virtual particle calorimeter plugin.
     *
     * (virtually) propagates and collects particles to infinite distance.
     *
     */
    template<class ParticlesType>
    class ParticleCalorimeter : public plugins::multi::IInstance
    {
        using DBufCalorimeter = pmacc::DeviceBuffer<float_X, DIM3>;
        using HBufCalorimeter = pmacc::HostBuffer<float_X, DIM3>;

    public:
        using MyCalorimeterFunctor = CalorimeterFunctor<typename DBufCalorimeter::DataBoxType>;

    private:
        std::shared_ptr<MyCalorimeterFunctor> calorimeterFunctor;

        std::unique_ptr<pmacc::mpi::MPIReduce> allGPU_reduce;

        template<typename T>
        std::vector<T> twoDimensional(std::vector<T> vector)
        {
            if(this->numBinsEnergy == 1)
            {
                vector.erase(vector.begin());
            }
            return vector;
        };

        void writeMeta(
            ::openPMD::Series& series,
            ::openPMD::Mesh& mesh,
            ::openPMD::MeshRecordComponent& dataset,
            uint32_t currentStep)
        {
            dataset.setAttribute<float_X>("maxPitch[deg]", maxPitch_deg);
            dataset.setAttribute<float_X>("maxYaw[deg]", maxYaw_deg);
            dataset.setAttribute<float_64>("posPitch[deg]", posPitch_deg);
            dataset.setAttribute<float_64>("posYaw[deg]", posYaw_deg);

            openPMD::WriteMeta writeMeta;
            writeMeta(
                series,
                series.iterations[currentStep],
                currentStep,
                /* writeFieldMeta =  */ false,
                /* writeParticleMeta =  */ false,
                /* writeToLog =  */ false);
            openPMD::SetMeshAttributes setMeshAttributes(currentStep);

            // override some attributes according to the calorimeter plugin
            if(this->numBinsEnergy > 1)
            {
                const float_64 minEnergy_SI = this->minEnergy * UNIT_ENERGY;
                const float_64 maxEnergy_SI = this->maxEnergy * UNIT_ENERGY;
                const float_64 minEnergy_keV = minEnergy_SI * UNITCONV_Joule_to_keV;
                const float_64 maxEnergy_keV = maxEnergy_SI * UNITCONV_Joule_to_keV;

                dataset.setAttribute<float_64>("minEnergy[keV]", minEnergy_keV);
                dataset.setAttribute<float_64>("maxEnergy[keV]", maxEnergy_keV);
                dataset.setAttribute<bool>("logScale", this->logScale);

                setMeshAttributes.m_axisLabels = {"energy bin", "posPitch", "posYaw"}; // z, y, x
                setMeshAttributes.m_gridGlobalOffset = {minEnergy_keV, posPitch_deg, posYaw_deg};
                setMeshAttributes.m_gridSpacing
                    = {// no constant value for grid spacing if the energy bins are on a log scale
                       this->logScale ? 0 : float_X(maxEnergy_keV - minEnergy_keV) / this->numBinsEnergy,
                       float_X(maxPitch_deg - posPitch_deg) / dataset.getExtent()[1],
                       float_X(maxYaw_deg - posYaw_deg) / dataset.getExtent()[2]};
            }
            else
            {
                setMeshAttributes.m_axisLabels = {"posPitch", "posYaw"}; // y, x
                setMeshAttributes.m_gridGlobalOffset = {posPitch_deg, posYaw_deg};
                setMeshAttributes.m_gridSpacing
                    = {float_X(maxPitch_deg - posPitch_deg) / dataset.getExtent()[1],
                       float_X(maxYaw_deg - posYaw_deg) / dataset.getExtent()[2]};
            }
            constexpr float_64 unitSI = particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE * UNIT_ENERGY;
            setMeshAttributes.m_unitSI = unitSI;
            setMeshAttributes.m_unitDimension // Joule
                = {{::openPMD::UnitDimension::M, 1},
                   {::openPMD::UnitDimension::L, 2},
                   {::openPMD::UnitDimension::T, -2}};
            // gridUnitDimension coming with openPMD 2.0, until then, this is somewhat custom
            setMeshAttributes.m_gridUnitSI = 1;

            setMeshAttributes(mesh)(dataset);
            // Custom geometries unsupported
            // mesh.setAttribute("geometry", "Calorimeter");
        }

    public:
        void restart(uint32_t restartStep, const std::string& restartDirectory) override
        {
            HBufCalorimeter hBufLeftParsCalorimeter(this->dBufLeftParsCalorimeter->getDataSpace());

            pmacc::GridController<simDim>& gridCon = pmacc::Environment<simDim>::get().GridController();
            pmacc::CommunicatorMPI<simDim>& comm = gridCon.getCommunicator();
            uint32_t rank = comm.getRank();

            if(rank == 0)
            {
                std::stringstream filename;
                filename << restartDirectory << "/" << this->foldername << "/" << filenamePrefix << "_%T."
                         << filenameExtension;

                ::openPMD::Series series(filename.str(), ::openPMD::Access::READ_ONLY);

                auto dataset = series.iterations[restartStep]
                                   .meshes[this->leftParticlesDatasetName][::openPMD::RecordComponent::SCALAR];

                ::openPMD::Extent extent = dataset.getExtent();
                ::openPMD::Offset offset(extent.size(), 0);
                dataset.loadChunk(
                    std::shared_ptr<float_X>{hBufLeftParsCalorimeter.getPointer(), [](auto const*) {}},
                    offset,
                    extent);

                series.iterations[restartStep].close();

                /* rank 0 divides and distributes the calorimeter to all ranks in equal parts */
                uint32_t numRanks = gridCon.getGlobalSize();

                /** @todo use foreach to walk over all elements, we can do this for loop only because we know that host
                 * buffer has no pitch
                 */
                auto* dataPtr = hBufLeftParsCalorimeter.getPointer();
                for(size_t i = 0u; i < hBufLeftParsCalorimeter.getCurrentSize(); ++i)
                    dataPtr[i] /= float_X(numRanks);
            }

            // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
            eventSystem::getTransactionEvent().waitForFinished();
            MPI_CHECK(MPI_Bcast(
                hBufLeftParsCalorimeter.getPointer(),
                hBufLeftParsCalorimeter.getCurrentSize() * sizeof(float_X),
                MPI_CHAR,
                0, /* rank 0 */
                comm.getMPIComm()));

            this->dBufLeftParsCalorimeter->copyFrom(hBufLeftParsCalorimeter);
        }


        void checkpoint(uint32_t currentStep, const std::string& checkpointDirectory) override
        {
            /*
             * Create folder for openPMD checkpoint files.
             * openPMD would also do it automatically, but let's keep things explicit.
             */
            Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(
                checkpointDirectory + "/" + this->foldername);
            auto dataSize = this->dBufLeftParsCalorimeter->getDataSpace();
            HBufCalorimeter hBufLeftParsCalorimeter(dataSize);
            HBufCalorimeter hBufTotal(dataSize);

            hBufLeftParsCalorimeter.copyFrom(*this->dBufLeftParsCalorimeter);

            /* mpi reduce */

            /* mpi reduce */
            (*allGPU_reduce)(
                pmacc::math::operation::Add(),
                hBufTotal.getPointer(),
                hBufLeftParsCalorimeter.getPointer(),
                hBufTotal.getCurrentSize(),
                mpi::reduceMethods::Reduce());

            if(!this->allGPU_reduce->hasResult(mpi::reduceMethods::Reduce()))
                return;

            std::stringstream filename;
            filename << checkpointDirectory << "/" << this->foldername << "/" << filenamePrefix << "_%T."
                     << filenameExtension;
            ::openPMD::Series series(filename.str(), ::openPMD::Access::CREATE);

            auto dataSize64Bit = precisionCast<size_t>(dataSize);
            auto bufferOffset = twoDimensional(::openPMD::Extent{0, 0, 0});
            auto bufferExtent
                = twoDimensional(::openPMD::Extent{dataSize64Bit.z(), dataSize64Bit.y(), dataSize64Bit.x()});

            auto iteration = series.iterations[currentStep];
            auto mesh = iteration.meshes[this->leftParticlesDatasetName];
            auto dataset = mesh[::openPMD::RecordComponent::SCALAR];

            dataset.resetDataset({::openPMD::determineDatatype<float_X>(), bufferExtent});
            dataset.storeChunk(
                std::shared_ptr<float_X>{hBufTotal.getPointer(), [](auto const*) {}},
                bufferOffset,
                bufferExtent);
            writeMeta(series, mesh, dataset, currentStep);
            iteration.close();
        }

    private:
        void initPlugin()
        {
            namespace pm = pmacc::math;

            if(!(this->openingYaw_deg > float_X(0.0) && this->openingYaw_deg <= float_X(360.0)))
            {
                std::stringstream msg;
                msg << "[Plugin] [" << m_help->getOptionPrefix() << "] openingYaw has to be within (0, 360]."
                    << std::endl;
                throw std::runtime_error(msg.str());
            }
            if(!(this->openingPitch_deg > float_X(0.0) && this->openingPitch_deg <= float_X(180.0)))
            {
                std::stringstream msg;
                msg << "[Plugin] [" << m_help->getOptionPrefix() << "] openingPitch has to be within (0, 180]."
                    << std::endl;
                throw std::runtime_error(msg.str());
            }
            if(this->minEnergy < float_X(0.0))
            {
                std::stringstream msg;
                msg << "[Plugin] [" << m_help->getOptionPrefix() << "] minEnergy can not be negative." << std::endl;
                throw std::runtime_error(msg.str());
            }
            if(this->logScale && this->minEnergy == float_X(0.0))
            {
                std::stringstream msg;
                msg << "[Plugin] [" << m_help->getOptionPrefix()
                    << "] minEnergy can not be zero in logarithmic scaling." << std::endl;
                throw std::runtime_error(msg.str());
            }
            if(this->numBinsEnergy > 1 && this->maxEnergy <= this->minEnergy)
            {
                std::stringstream msg;
                msg << "[Plugin] [" << m_help->getOptionPrefix() << "] minEnergy has to be less than maxEnergy."
                    << std::endl;
                throw std::runtime_error(msg.str());
            }

            this->maxYaw_deg = float_X(0.5) * this->openingYaw_deg;
            this->maxPitch_deg = float_X(0.5) * this->openingPitch_deg;
            /* convert units */
            const float_64 minEnergy_SI = this->minEnergy * UNITCONV_keV_to_Joule;
            const float_64 maxEnergy_SI = this->maxEnergy * UNITCONV_keV_to_Joule;
            this->minEnergy = minEnergy_SI / UNIT_ENERGY;
            this->maxEnergy = maxEnergy_SI / UNIT_ENERGY;

            /* allocate memory buffers */
            auto detectorSize = DataSpace<DIM3>(this->numBinsYaw, this->numBinsPitch, this->numBinsEnergy);
            this->dBufCalorimeter = std::make_unique<DBufCalorimeter>(detectorSize);
            this->dBufLeftParsCalorimeter = std::make_unique<DBufCalorimeter>(detectorSize);
            this->hBufCalorimeter = std::make_unique<HBufCalorimeter>(detectorSize);
            this->hBufTotalCalorimeter = std::make_unique<HBufCalorimeter>(detectorSize);

            /* fill calorimeter for left particles with zero */
            this->dBufLeftParsCalorimeter->setValue(float_X(0.0));

            /* create mpi reduce algorithm */
            this->allGPU_reduce = std::make_unique<pmacc::mpi::MPIReduce>();
            this->allGPU_reduce->participate(true);

            /* calculate rotated calorimeter frame from posYaw_deg and posPitch_deg */
            constexpr float_64 radsInDegree = pmacc::math::Pi<float_64>::value / float_64(180.0);
            const float_64 posYaw_rad = this->posYaw_deg * radsInDegree;
            const float_64 posPitch_rad = this->posPitch_deg * radsInDegree;
            this->calorimeterFrameVecY = float3_X(
                math::sin(posYaw_rad) * math::cos(posPitch_rad),
                math::cos(posYaw_rad) * math::cos(posPitch_rad),
                math::sin(posPitch_rad));
            /* If the y-axis is pointing exactly up- or downwards we need to define the x-axis manually */
            if(math::abs(this->calorimeterFrameVecY.z()) == float_X(1.0))
            {
                this->calorimeterFrameVecX = float3_X(1.0, 0.0, 0.0);
            }
            else
            {
                /* choose `calorimeterFrameVecX` so that the roll is zero. */
                const float3_X vecUp(0.0, 0.0, -1.0);
                this->calorimeterFrameVecX = pmacc::math::cross(vecUp, this->calorimeterFrameVecY);
                /* normalize vector */
                this->calorimeterFrameVecX /= pmacc::math::l2norm(this->calorimeterFrameVecX);
            }
            this->calorimeterFrameVecZ = pmacc::math::cross(this->calorimeterFrameVecX, this->calorimeterFrameVecY);

            /* create calorimeter functor instance */
            this->calorimeterFunctor = std::make_shared<MyCalorimeterFunctor>(
                this->maxYaw_deg * radsInDegree,
                this->maxPitch_deg * radsInDegree,
                this->numBinsYaw,
                this->numBinsPitch,
                this->numBinsEnergy,
                this->logScale ? pmacc::math::log10(this->minEnergy) : this->minEnergy,
                this->logScale ? pmacc::math::log10(this->maxEnergy) : this->maxEnergy,
                this->logScale,
                this->calorimeterFrameVecX,
                this->calorimeterFrameVecY,
                this->calorimeterFrameVecZ);

            /* create folder for openPMD files*/
            Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(this->foldername);

            // set how often the plugin should be executed while PIConGPU is running
            Environment<>::get().PluginConnector().setNotificationPeriod(this, m_help->notifyPeriod.get(m_id));
        }

        void writeToOpenPMDFile(uint32_t currentStep)
        {
            std::stringstream filename;
            filename << this->foldername << "/" << filenamePrefix << "_%T." << filenameExtension;
            ::openPMD::Series series(filename.str(), ::openPMD::Access::CREATE);

            auto offset = twoDimensional(::openPMD::Offset{0, 0, 0});

            auto dataSize = this->hBufTotalCalorimeter->getCurrentDataSpace();
            auto dataSize64Bit = precisionCast<size_t>(dataSize);
            auto extent = twoDimensional(::openPMD::Extent{dataSize64Bit.z(), dataSize64Bit.y(), dataSize64Bit.x()});

            auto mesh = series.iterations[currentStep].meshes["calorimeter"];
            auto calorimeter = mesh[::openPMD::RecordComponent::SCALAR];
            calorimeter.resetDataset({::openPMD::determineDatatype<float_X>(), extent});
            calorimeter.storeChunk(
                std::shared_ptr<float_X>{this->hBufTotalCalorimeter->getPointer(), [](auto const*) {}},
                std::move(offset),
                std::move(extent));

            // Write attributes
            writeMeta(series, mesh, calorimeter, currentStep);

            series.iterations[currentStep].close();
        }

    public:
        struct Help : public plugins::multi::IHelp
        {
            /** creates an instance
             *
             * @param help plugin defined help
             * @param id index of the plugin, range: [0;help->getNumPlugins())
             */
            std::shared_ptr<IInstance> create(
                std::shared_ptr<IHelp>& help,
                size_t const id,
                MappingDesc* cellDescription) override
            {
                return std::shared_ptr<IInstance>(new ParticleCalorimeter<ParticlesType>(help, id, cellDescription));
            }

            // find all valid filter for the current used species
            template<typename T>
            using Op = typename particles::traits::GenerateSolversIfSpeciesEligible<T, ParticlesType>::type;
            using EligibleFilters = pmacc::mp_flatten<pmacc::mp_transform<Op, particles::filter::AllParticleFilters>>;

            //! periodicity of computing the particle energy
            plugins::multi::Option<std::string> notifyPeriod = {"period", "enable plugin [for each n-th step]"};
            plugins::multi::Option<std::string> fileName = {"file", "output filename (prefix)"};
            plugins::multi::Option<std::string> filter = {"filter", "particle filter: "};
            plugins::multi::Option<std::string> extension
                = {"ext", "openPMD filename extension", openPMD::getDefaultExtension().c_str()};
            plugins::multi::Option<uint32_t> numBinsYaw = {"numBinsYaw", "number of bins for angle yaw.", 64};
            plugins::multi::Option<uint32_t> numBinsPitch = {"numBinsPitch", "number of bins for angle pitch.", 64};
            plugins::multi::Option<uint32_t> numBinsEnergy
                = {"numBinsEnergy", "number of bins for the energy spectrum. Disabled by default.", 1};
            plugins::multi::Option<float_X> minEnergy = {"minEnergy", "minimal detectable energy in keV.", 0.0};
            plugins::multi::Option<float_X> maxEnergy = {"maxEnergy", "maximal detectable energy in keV.", 1.0e3};
            plugins::multi::Option<uint32_t> logScale = {"logScale", "enable logarithmic energy scale.", 0};
            plugins::multi::Option<float_X> openingYaw
                = {"openingYaw", "opening angle yaw in degrees. 0 <= x <= 360.", 360.0};
            plugins::multi::Option<float_X> openingPitch
                = {"openingPitch", "opening angle pitch in degrees. 0 <= x <= 180.", 180.0};
            plugins::multi::Option<float_64> posYaw
                = {"posYaw", "yaw coordinate of calorimeter position in degrees. Defaults to +y direction.", 0.0};
            plugins::multi::Option<float_64> posPitch
                = {"posPitch", "pitch coordinate of calorimeter position in degrees. Defaults to +y direction.", 0.0};

            //! string list with all possible particle filters
            std::string concatenatedFilterNames;
            std::vector<std::string> allowedFilters;

            ///! method used by plugin controller to get --help description
            void registerHelp(
                boost::program_options::options_description& desc,
                std::string const& masterPrefix = std::string{}) override
            {
                meta::ForEach<EligibleFilters, plugins::misc::AppendName<boost::mpl::_1>> getEligibleFilterNames;
                getEligibleFilterNames(allowedFilters);

                concatenatedFilterNames = plugins::misc::concatenateToString(allowedFilters, ", ");

                notifyPeriod.registerHelp(desc, masterPrefix + prefix);
                fileName.registerHelp(desc, masterPrefix + prefix);
                extension.registerHelp(desc, masterPrefix + prefix);
                filter.registerHelp(desc, masterPrefix + prefix, std::string("[") + concatenatedFilterNames + "]");
                numBinsYaw.registerHelp(desc, masterPrefix + prefix);
                numBinsPitch.registerHelp(desc, masterPrefix + prefix);
                numBinsEnergy.registerHelp(desc, masterPrefix + prefix);
                minEnergy.registerHelp(desc, masterPrefix + prefix);
                maxEnergy.registerHelp(desc, masterPrefix + prefix);
                logScale.registerHelp(desc, masterPrefix + prefix);
                openingYaw.registerHelp(desc, masterPrefix + prefix);
                openingPitch.registerHelp(desc, masterPrefix + prefix);
                posYaw.registerHelp(desc, masterPrefix + prefix);
                posPitch.registerHelp(desc, masterPrefix + prefix);
            }

            void expandHelp(
                boost::program_options::options_description& desc,
                std::string const& masterPrefix = std::string{}) override
            {
            }


            void validateOptions() override
            {
                if(notifyPeriod.size() != fileName.size())
                    throw std::runtime_error(
                        name + ": parameter file and period are not used the same number of times");

                if(notifyPeriod.size() != filter.size())
                    throw std::runtime_error(
                        name + ": parameter filter and period are not used the same number of times");

                // check if user passed filter name are valid
                for(auto const& filterName : filter)
                {
                    if(std::find(allowedFilters.begin(), allowedFilters.end(), filterName) == allowedFilters.end())
                    {
                        throw std::runtime_error(name + ": unknown filter '" + filterName + "'");
                    }
                }
            }

            size_t getNumPlugins() const override
            {
                return notifyPeriod.size();
            }

            std::string getDescription() const override
            {
                return description;
            }

            std::string getOptionPrefix() const
            {
                return prefix;
            }

            std::string getName() const override
            {
                return name;
            }

            std::string const name = "ParticleCalorimeter";
            //! short description of the plugin
            std::string const description = "(virtually) propagates and collects particles to infinite distance";
            //! prefix used for command line arguments
            std::string const prefix = ParticlesType::FrameType::getName() + std::string("_calorimeter");
        };

        static std::shared_ptr<plugins::multi::IHelp> getHelp()
        {
            return std::shared_ptr<plugins::multi::IHelp>(new Help{});
        }

        ParticleCalorimeter(
            std::shared_ptr<plugins::multi::IHelp>& help,
            size_t const id,
            MappingDesc* cellDescription)
            : m_help(std::static_pointer_cast<Help>(help))
            , m_id(id)
            , m_cellDescription(cellDescription)
            , leftParticlesDatasetName("calorimeterLeftParticles")
        {
            foldername = m_help->getOptionPrefix() + "/";
            filenamePrefix = m_help->getOptionPrefix() + "_" + m_help->filter.get(m_id);
            filenameExtension = m_help->extension.get(m_id);
            numBinsYaw = m_help->numBinsYaw.get(m_id);
            numBinsPitch = m_help->numBinsPitch.get(m_id);
            numBinsEnergy = m_help->numBinsEnergy.get(m_id);
            minEnergy = m_help->minEnergy.get(m_id);
            maxEnergy = m_help->maxEnergy.get(m_id);
            logScale = m_help->logScale.get(m_id);
            openingYaw_deg = m_help->openingYaw.get(m_id);
            openingPitch_deg = m_help->openingPitch.get(m_id);
            posYaw_deg = m_help->posYaw.get(m_id);
            posPitch_deg = m_help->posPitch.get(m_id);
            notifyPeriod = m_help->notifyPeriod.get(m_id);

            initPlugin();
        }

        void notify(uint32_t currentStep) override
        {
            /* initialize calorimeter with already detected particles */
            this->dBufCalorimeter->copyFrom(*this->dBufLeftParsCalorimeter);

            /* data is written to dBufCalorimeter */
            this->calorimeterFunctor->setCalorimeterData(this->dBufCalorimeter->getDataBox());

            /* create kernel functor instance */
            DataConnector& dc = Environment<>::get().DataConnector();
            auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName());

            auto const mapper = makeAreaMapper<CORE + BORDER>(*this->m_cellDescription);
            auto const grid = mapper.getGridDim();

            // In this version we process all particles, so internal area = total domain
            SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
            auto beginInternalCellsLocal = pmacc::DataSpace<simDim>::create(0);
            auto endInternalCellsLocal = beginInternalCellsLocal + subGrid.getLocalDomain().size;

            auto workerCfg = lockstep::makeWorkerCfg<ParticlesType::FrameType::frameSize>();

            auto kernel = PMACC_LOCKSTEP_KERNEL(KernelParticleCalorimeter{}, workerCfg)(grid);
            auto unaryKernel = std::bind(
                kernel,
                particles->getDeviceParticlesBox(),
                *this->calorimeterFunctor,
                mapper,
                beginInternalCellsLocal,
                endInternalCellsLocal,
                std::placeholders::_1);

            meta::ForEach<typename Help::EligibleFilters, plugins::misc::ExecuteIfNameIsEqual<boost::mpl::_1>>{}(
                m_help->filter.get(m_id),
                currentStep,
                unaryKernel);

            /* copy to host */
            this->hBufCalorimeter->copyFrom(*this->dBufCalorimeter);

            /* mpi reduce */
            (*allGPU_reduce)(
                pmacc::math::operation::Add(),
                this->hBufTotalCalorimeter->getPointer(),
                this->hBufCalorimeter->getPointer(),
                this->hBufCalorimeter->getCurrentSize(),
                mpi::reduceMethods::Reduce());

            if(!this->allGPU_reduce->hasResult(mpi::reduceMethods::Reduce()))
                return;

            this->writeToOpenPMDFile(currentStep);
        }

        void onParticleLeave(const std::string& speciesName, int32_t direction) override
        {
            if(this->notifyPeriod.empty())
                return;
            if(speciesName != ParticlesType::FrameType::getName())
                return;

            /* data is written to dBufLeftParsCalorimeter */
            this->calorimeterFunctor->setCalorimeterData(this->dBufLeftParsCalorimeter->getDataBox());

            DataConnector& dc = Environment<>::get().DataConnector();
            auto particles = dc.get<ParticlesType>(speciesName);

            auto mapperFactory = particles::boundary::getMapperFactory(*particles, direction);
            auto const mapper = mapperFactory(*this->m_cellDescription);
            auto grid = mapper.getGridDim();

            /* Here we only process the particles that just crossed the boundary,
             * so the active area for the kernel is the outside area wrt the boundary
             */
            pmacc::DataSpace<simDim> beginExternalCellsTotal, endExternalCellsTotal;
            particles::boundary::getExternalCellsTotal(
                *particles,
                direction,
                &beginExternalCellsTotal,
                &endExternalCellsTotal);
            SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
            pmacc::DataSpace<simDim> shiftTotaltoLocal
                = subGrid.getGlobalDomain().offset + subGrid.getLocalDomain().offset;
            auto const beginExternalCellsLocal = beginExternalCellsTotal - shiftTotaltoLocal;
            auto const endExternalCellsLocal = endExternalCellsTotal - shiftTotaltoLocal;

            auto workerCfg = lockstep::makeWorkerCfg<ParticlesType::FrameType::frameSize>();

            auto kernel = PMACC_LOCKSTEP_KERNEL(KernelParticleCalorimeter{}, workerCfg)(grid);
            auto unaryKernel = std::bind(
                kernel,
                particles->getDeviceParticlesBox(),
                (MyCalorimeterFunctor) * this->calorimeterFunctor,
                mapper,
                beginExternalCellsLocal,
                endExternalCellsLocal,
                std::placeholders::_1);

            meta::ForEach<typename Help::EligibleFilters, plugins::misc::ExecuteIfNameIsEqual<boost::mpl::_1>>{}(
                m_help->filter.get(m_id),
                Environment<>::get().SimulationDescription().getCurrentStep(),
                unaryKernel);
        }

    private:
        std::shared_ptr<Help> m_help;
        size_t m_id;
        std::string foldername;
        std::string filenamePrefix;
        std::string filenameExtension;
        MappingDesc* m_cellDescription;
        std::ofstream outFile;
        const std::string leftParticlesDatasetName;
        std::string notifyPeriod;

        uint32_t numBinsYaw;
        uint32_t numBinsPitch;
        uint32_t numBinsEnergy;
        float_X minEnergy;
        float_X maxEnergy;
        bool logScale;
        float_X openingYaw_deg;
        float_X openingPitch_deg;
        float_X maxYaw_deg;
        float_X maxPitch_deg;

        float_64 posYaw_deg;
        float_64 posPitch_deg;

        //! Rotated calorimeter frame
        float3_X calorimeterFrameVecX;
        float3_X calorimeterFrameVecY;
        float3_X calorimeterFrameVecZ;

        //! device calorimeter buffer for a single gpu
        std::unique_ptr<DBufCalorimeter> dBufCalorimeter;
        //! device calorimeter buffer for all particles which have left the simulation volume
        std::unique_ptr<DBufCalorimeter> dBufLeftParsCalorimeter;
        //! host calorimeter buffer for a single mpi rank
        std::unique_ptr<HBufCalorimeter> hBufCalorimeter;
        //! host calorimeter buffer for summation of all mpi ranks
        std::unique_ptr<HBufCalorimeter> hBufTotalCalorimeter;
    };

    namespace particles
    {
        namespace traits
        {
            template<typename T_Species, typename T_UnspecifiedSpecies>
            struct SpeciesEligibleForSolver<T_Species, ParticleCalorimeter<T_UnspecifiedSpecies>>
            {
                using FrameType = typename T_Species::FrameType;

                // this plugin needs at least the weighting and momentum attributes
                using RequiredIdentifiers = MakeSeq_t<weighting, momentum>;

                using SpeciesHasIdentifiers =
                    typename pmacc::traits::HasIdentifiers<FrameType, RequiredIdentifiers>::type;

                // and also a mass ratio for energy calculation from momentum
                using SpeciesHasFlags = typename pmacc::traits::HasFlag<FrameType, massRatio<>>::type;

                using type = pmacc::mp_and<SpeciesHasIdentifiers, SpeciesHasFlags>;
            };
        } // namespace traits
    } // namespace particles
} // namespace picongpu
