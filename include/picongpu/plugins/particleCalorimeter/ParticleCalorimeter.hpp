/* Copyright 2016-2021 Heiko Burau, Rene Widera
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

#include "ParticleCalorimeterFunctors.hpp"
#include "ParticleCalorimeter.kernel"

#include "picongpu/traits/PICToSplash.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/multi/multi.hpp"
#include "picongpu/plugins/misc/misc.hpp"

#include <pmacc/cuSTL/container/DeviceBuffer.hpp>
#include <pmacc/cuSTL/container/HostBuffer.hpp>
#include <pmacc/cuSTL/algorithm/kernel/Foreach.hpp>
#include <pmacc/cuSTL/cursor/MultiIndexCursor.hpp>
#include <pmacc/cuSTL/algorithm/mpi/Reduce.hpp>
#include <pmacc/cuSTL/algorithm/host/Foreach.hpp>
#include <pmacc/particles/policies/ExchangeParticles.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/algorithms/math.hpp>
#include <pmacc/cuSTL/algorithm/functor/Add.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/traits/HasIdentifiers.hpp>
#include <pmacc/traits/HasFlag.hpp>

#include <splash/splash.h>
#include <boost/filesystem.hpp>
#include <boost/mpl/and.hpp>
#include <boost/shared_ptr.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>


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
    class ParticleCalorimeter : public plugins::multi::ISlave
    {
        typedef pmacc::container::DeviceBuffer<float_X, DIM3> DBufCalorimeter;
        typedef pmacc::container::HostBuffer<float_X, DIM3> HBufCalorimeter;


        template<typename T_Type>
        struct DivideInPlace
        {
            using Type = T_Type;
            const Type divisor;

            DivideInPlace(const Type& divisor) : divisor(divisor)
            {
            }

            template<typename T_Acc>
            HDINLINE void operator()(T_Acc const&, T_Type& val) const
            {
                val = val / this->divisor;
            }
        };

    public:
        typedef CalorimeterFunctor<typename DBufCalorimeter::Cursor> MyCalorimeterFunctor;

    private:
        typedef boost::shared_ptr<MyCalorimeterFunctor> MyCalorimeterFunctorPtr;
        MyCalorimeterFunctorPtr calorimeterFunctor;

        typedef boost::shared_ptr<pmacc::algorithm::mpi::Reduce<simDim>> AllGPU_reduce;
        AllGPU_reduce allGPU_reduce;

    public:
        void restart(uint32_t restartStep, const std::string& restartDirectory)
        {
            HBufCalorimeter hBufLeftParsCalorimeter(this->dBufLeftParsCalorimeter->size());

            pmacc::GridController<simDim>& gridCon = pmacc::Environment<simDim>::get().GridController();
            pmacc::CommunicatorMPI<simDim>& comm = gridCon.getCommunicator();
            uint32_t rank = comm.getRank();

            if(rank == 0)
            {
                splash::SerialDataCollector hdf5DataFile(1);
                splash::DataCollector::FileCreationAttr fAttr;

                splash::DataCollector::initFileCreationAttr(fAttr);
                fAttr.fileAccType = splash::DataCollector::FAT_READ;

                std::stringstream filename;
                filename << restartDirectory << "/" << (this->foldername + "/" + filenamePrefix) << "_" << restartStep;

                hdf5DataFile.open(filename.str().c_str(), fAttr);

                splash::Dimensions dimensions;

                hdf5DataFile.read(
                    restartStep,
                    this->leftParticlesDatasetName.c_str(),
                    dimensions,
                    &(*hBufLeftParsCalorimeter.origin()));

                hdf5DataFile.close();

                /* rank 0 divides and distributes the calorimeter to all ranks in equal parts */
                uint32_t numRanks = gridCon.getGlobalSize();
                // get a host accelerator
                auto hostDev = cupla::manager::Device<cupla::AccHost>::get().device();
                pmacc::algorithm::host::Foreach()(
                    hostDev,
                    hBufLeftParsCalorimeter.zone(),
                    hBufLeftParsCalorimeter.origin(),
                    DivideInPlace<float_X>(float_X(numRanks)));
            }

            // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
            __getTransactionEvent().waitForFinished();
            MPI_Bcast(
                &(*hBufLeftParsCalorimeter.origin()),
                hBufLeftParsCalorimeter.size().productOfComponents() * sizeof(float_X),
                MPI_CHAR,
                0, /* rank 0 */
                comm.getMPIComm());

            *this->dBufLeftParsCalorimeter = hBufLeftParsCalorimeter;
        }


        void checkpoint(uint32_t currentStep, const std::string& checkpointDirectory)
        {
            /* create folder for hdf5 checkpoint files*/
            Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(
                checkpointDirectory + "/" + this->foldername);
            HBufCalorimeter hBufLeftParsCalorimeter(this->dBufLeftParsCalorimeter->size());
            HBufCalorimeter hBufTotal(hBufLeftParsCalorimeter.size());

            hBufLeftParsCalorimeter = *this->dBufLeftParsCalorimeter;

            /* mpi reduce */
            (*this->allGPU_reduce)(hBufTotal, hBufLeftParsCalorimeter, pmacc::algorithm::functor::Add{});
            if(!this->allGPU_reduce->root())
                return;

            splash::SerialDataCollector hdf5DataFile(1);
            splash::DataCollector::FileCreationAttr fAttr;

            splash::DataCollector::initFileCreationAttr(fAttr);

            std::stringstream filename;
            filename << checkpointDirectory << "/" << (this->foldername + "/" + filenamePrefix) << "_" << currentStep;

            hdf5DataFile.open(filename.str().c_str(), fAttr);

            typename PICToSplash<float_X>::type SplashTypeX;

            splash::Dimensions bufferSize(hBufTotal.size().x(), hBufTotal.size().y(), hBufTotal.size().z());

            /* if there is only one energy bin, omit the energy axis */
            uint32_t dimension = this->numBinsEnergy == 1 ? DIM2 : DIM3;
            hdf5DataFile.write(
                currentStep,
                SplashTypeX,
                dimension,
                splash::Selection(bufferSize),
                this->leftParticlesDatasetName.c_str(),
                &(*hBufTotal.origin()));

            hdf5DataFile.close();
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
            this->dBufCalorimeter = new DBufCalorimeter(this->numBinsYaw, this->numBinsPitch, this->numBinsEnergy);
            this->dBufLeftParsCalorimeter = new DBufCalorimeter(this->dBufCalorimeter->size());
            this->hBufCalorimeter = new HBufCalorimeter(this->dBufCalorimeter->size());
            this->hBufTotalCalorimeter = new HBufCalorimeter(this->dBufCalorimeter->size());

            /* fill calorimeter for left particles with zero */
            this->dBufLeftParsCalorimeter->assign(float_X(0.0));

            /* create mpi reduce algorithm */
            pmacc::GridController<simDim>& con = pmacc::Environment<simDim>::get().GridController();
            pm::Size_t<simDim> gpuDim = (pm::Size_t<simDim>) con.getGpuNodes();
            zone::SphericZone<simDim> zone_allGPUs(gpuDim);
            this->allGPU_reduce = AllGPU_reduce(new pmacc::algorithm::mpi::Reduce<simDim>(zone_allGPUs));

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
                this->calorimeterFrameVecX /= math::abs(this->calorimeterFrameVecX);
            }
            this->calorimeterFrameVecZ = pmacc::math::cross(this->calorimeterFrameVecX, this->calorimeterFrameVecY);

            /* create calorimeter functor instance */
            this->calorimeterFunctor = MyCalorimeterFunctorPtr(new MyCalorimeterFunctor(
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
                this->calorimeterFrameVecZ));

            /* create folder for hdf5 files*/
            Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(this->foldername);

            // set how often the plugin should be executed while PIConGPU is running
            Environment<>::get().PluginConnector().setNotificationPeriod(this, m_help->notifyPeriod.get(m_id));
        }

        void writeToHDF5File(uint32_t currentStep)
        {
            splash::SerialDataCollector hdf5DataFile(1);
            splash::DataCollector::FileCreationAttr fAttr;

            splash::DataCollector::initFileCreationAttr(fAttr);

            std::stringstream filename;
            filename << this->foldername << "/" << filenamePrefix << "_" << currentStep;

            hdf5DataFile.open(filename.str().c_str(), fAttr);

            typename PICToSplash<float_X>::type SplashTypeX;
            typename PICToSplash<float_64>::type SplashType64;
            typename PICToSplash<bool>::type SplashTypeBool;

            splash::Dimensions bufferSize(
                this->hBufTotalCalorimeter->size().x(),
                this->hBufTotalCalorimeter->size().y(),
                this->hBufTotalCalorimeter->size().z());

            hdf5DataFile.write(
                currentStep,
                SplashTypeX,
                this->numBinsEnergy == 1 ? DIM2 : DIM3,
                splash::Selection(bufferSize),
                "calorimeter",
                &(*this->hBufTotalCalorimeter->origin()));

            const float_64 unitSI = particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE * UNIT_ENERGY;

            hdf5DataFile.writeAttribute(currentStep, SplashType64, "calorimeter", "unitSI", &unitSI);

            hdf5DataFile.writeAttribute(currentStep, SplashType64, "calorimeter", "posYaw[deg]", &posYaw_deg);

            hdf5DataFile.writeAttribute(currentStep, SplashType64, "calorimeter", "posPitch[deg]", &posPitch_deg);

            hdf5DataFile.writeAttribute(currentStep, SplashTypeX, "calorimeter", "maxYaw[deg]", &this->maxYaw_deg);

            hdf5DataFile.writeAttribute(currentStep, SplashTypeX, "calorimeter", "maxPitch[deg]", &this->maxPitch_deg);

            if(this->numBinsEnergy > 1)
            {
                const float_64 minEnergy_SI = this->minEnergy * UNIT_ENERGY;
                const float_64 maxEnergy_SI = this->maxEnergy * UNIT_ENERGY;
                const float_64 minEnergy_keV = minEnergy_SI * UNITCONV_Joule_to_keV;
                const float_64 maxEnergy_keV = maxEnergy_SI * UNITCONV_Joule_to_keV;

                hdf5DataFile
                    .writeAttribute(currentStep, SplashType64, "calorimeter", "minEnergy[keV]", &minEnergy_keV);

                hdf5DataFile
                    .writeAttribute(currentStep, SplashType64, "calorimeter", "maxEnergy[keV]", &maxEnergy_keV);

                hdf5DataFile.writeAttribute(currentStep, SplashTypeBool, "calorimeter", "logScale", &this->logScale);
            }

            hdf5DataFile.close();
        }

    public:
        struct Help : public plugins::multi::IHelp
        {
            /** creates an instance of ISlave
             *
             * @tparam T_Slave type of the interface implementation (must inherit from ISlave)
             * @param help plugin defined help
             * @param id index of the plugin, range: [0;help->getNumPlugins())
             */
            std::shared_ptr<ISlave> create(std::shared_ptr<IHelp>& help, size_t const id, MappingDesc* cellDescription)
            {
                return std::shared_ptr<ISlave>(new ParticleCalorimeter<ParticlesType>(help, id, cellDescription));
            }

            // find all valid filter for the current used species
            using EligibleFilters = typename MakeSeqFromNestedSeq<typename bmpl::transform<
                particles::filter::AllParticleFilters,
                particles::traits::GenerateSolversIfSpeciesEligible<bmpl::_1, ParticlesType>>::type>::type;

            //! periodicity of computing the particle energy
            plugins::multi::Option<std::string> notifyPeriod = {"period", "enable plugin [for each n-th step]"};
            plugins::multi::Option<std::string> fileName = {"file", "output filename (prefix)"};
            plugins::multi::Option<std::string> filter = {"filter", "particle filter: "};
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
                std::string const& masterPrefix = std::string{})
            {
                meta::ForEach<EligibleFilters, plugins::misc::AppendName<bmpl::_1>> getEligibleFilterNames;
                getEligibleFilterNames(allowedFilters);

                concatenatedFilterNames = plugins::misc::concatenateToString(allowedFilters, ", ");

                notifyPeriod.registerHelp(desc, masterPrefix + prefix);
                fileName.registerHelp(desc, masterPrefix + prefix);
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
                std::string const& masterPrefix = std::string{})
            {
            }


            void validateOptions()
            {
                if(notifyPeriod.size() != fileName.size())
                    throw std::runtime_error(
                        name + ": parameter fileName and period are not used the same number of times");

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

            size_t getNumPlugins() const
            {
                return notifyPeriod.size();
            }

            std::string getDescription() const
            {
                return description;
            }

            std::string getOptionPrefix() const
            {
                return prefix;
            }

            std::string getName() const
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
            , dBufCalorimeter(nullptr)
            , dBufLeftParsCalorimeter(nullptr)
            , hBufCalorimeter(nullptr)
            , hBufTotalCalorimeter(nullptr)
        {
            foldername = m_help->getOptionPrefix() + "/" + m_help->filter.get(m_id);
            filenamePrefix
                = m_help->getOptionPrefix() + "_" + m_help->fileName.get(m_id) + "_" + m_help->filter.get(m_id);
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

            initPlugin();
        }

        virtual ~ParticleCalorimeter()
        {
            __delete(this->dBufCalorimeter);
            __delete(this->dBufLeftParsCalorimeter);
            __delete(this->hBufCalorimeter);
            __delete(this->hBufTotalCalorimeter);
        }


        void notify(uint32_t currentStep)
        {
            /* initialize calorimeter with already detected particles */
            *this->dBufCalorimeter = *this->dBufLeftParsCalorimeter;

            /* data is written to dBufCalorimeter */
            this->calorimeterFunctor->setCalorimeterCursor(this->dBufCalorimeter->origin());

            /* create kernel functor instance */
            DataConnector& dc = Environment<>::get().DataConnector();
            auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName(), true);

            AreaMapping<CORE + BORDER, MappingDesc> const mapper(*this->m_cellDescription);
            auto const grid = mapper.getGridDim();

            constexpr uint32_t numWorkers
                = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

            auto kernel = PMACC_KERNEL(KernelParticleCalorimeter<numWorkers>{})(grid, numWorkers);
            auto unaryKernel = std::bind(
                kernel,
                particles->getDeviceParticlesBox(),
                *this->calorimeterFunctor,
                mapper,
                std::placeholders::_1);

            meta::ForEach<typename Help::EligibleFilters, plugins::misc::ExecuteIfNameIsEqual<bmpl::_1>>{}(
                m_help->filter.get(m_id),
                currentStep,
                unaryKernel);

            dc.releaseData(ParticlesType::FrameType::getName());

            /* copy to host */
            *this->hBufCalorimeter = *this->dBufCalorimeter;

            /* mpi reduce */
            (*this->allGPU_reduce)(
                *this->hBufTotalCalorimeter,
                *this->hBufCalorimeter,
                pmacc::algorithm::functor::Add{});
            if(!this->allGPU_reduce->root())
                return;

            this->writeToHDF5File(currentStep);
        }

        void onParticleLeave(const std::string& speciesName, int32_t direction)
        {
            if(this->notifyPeriod.empty())
                return;
            if(speciesName != ParticlesType::FrameType::getName())
                return;

            /* data is written to dBufLeftParsCalorimeter */
            this->calorimeterFunctor->setCalorimeterCursor(this->dBufLeftParsCalorimeter->origin());

            ExchangeMapping<GUARD, MappingDesc> mapper(*this->cellDescription, direction);
            auto grid = mapper.getGridDim();

            DataConnector& dc = Environment<>::get().DataConnector();
            auto particles = dc.get<ParticlesType>(speciesName, true);

            constexpr uint32_t numWorkers
                = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

            auto kernel = PMACC_KERNEL(KernelParticleCalorimeter<numWorkers>{})(grid, numWorkers);
            auto unaryKernel = std::bind(
                kernel,
                particles->getDeviceParticlesBox(),
                (MyCalorimeterFunctor) * this->calorimeterFunctor,
                mapper,
                std::placeholders::_1);

            meta::ForEach<typename Help::EligibleFilters, plugins::misc::ExecuteIfNameIsEqual<bmpl::_1>>{}(
                m_help->filter.get(m_id),
                Environment<>::get().SimulationDescription().getCurrentStep(),
                unaryKernel);

            dc.releaseData(speciesName);
        }

    private:
        std::shared_ptr<Help> m_help;
        size_t m_id;
        std::string foldername;
        std::string filenamePrefix;
        MappingDesc* m_cellDescription;
        std::ofstream outFile;
        const std::string leftParticlesDatasetName;

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
        DBufCalorimeter* dBufCalorimeter;
        //! device calorimeter buffer for all particles which have left the simulation volume
        DBufCalorimeter* dBufLeftParsCalorimeter;
        //! host calorimeter buffer for a single mpi rank
        HBufCalorimeter* hBufCalorimeter;
        //! host calorimeter buffer for summation of all mpi ranks
        HBufCalorimeter* hBufTotalCalorimeter;
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

                using type = typename bmpl::and_<SpeciesHasIdentifiers, SpeciesHasFlags>;
            };
        } // namespace traits
    } // namespace particles
} // namespace picongpu
