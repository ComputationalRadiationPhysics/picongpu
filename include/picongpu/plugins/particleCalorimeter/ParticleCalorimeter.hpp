/* Copyright 2016-2018 Heiko Burau
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
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"

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
class ParticleCalorimeter : public ISimulationPlugin
{
private:
    std::string name;
    std::string prefix;
    std::string foldername;
    std::string notifyPeriod;
    MappingDesc* cellDescription;
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

    /* Rotated calorimeter frame */
    float3_X calorimeterFrameVecX;
    float3_X calorimeterFrameVecY;
    float3_X calorimeterFrameVecZ;

    typedef pmacc::container::DeviceBuffer<float_X, DIM3> DBufCalorimeter;
    typedef pmacc::container::HostBuffer<float_X, DIM3> HBufCalorimeter;

    /* device calorimeter buffer for a single gpu */
    DBufCalorimeter* dBufCalorimeter;
    /* device calorimeter buffer for all particles which have left the simulation volume  */
    DBufCalorimeter* dBufLeftParsCalorimeter;
    /* host calorimeter buffer for a single mpi rank */
    HBufCalorimeter* hBufCalorimeter;
    /* host calorimeter buffer for summation of all mpi ranks */
    HBufCalorimeter* hBufTotalCalorimeter;

    template<typename T_Type>
    struct DivideInPlace
    {
        using Type = T_Type;
        const Type divisor;

        DivideInPlace( const Type& divisor ) : divisor( divisor ) {}

        template< typename T_Acc >
        HDINLINE void operator()( T_Acc const &, T_Type& val ) const
        {
            val = val / this->divisor;
        }
    };

public:
    typedef CalorimeterFunctor<typename DBufCalorimeter::Cursor> MyCalorimeterFunctor;
private:
    typedef boost::shared_ptr<MyCalorimeterFunctor> MyCalorimeterFunctorPtr;
    MyCalorimeterFunctorPtr calorimeterFunctor;

    typedef boost::shared_ptr<pmacc::algorithm::mpi::Reduce<simDim> > AllGPU_reduce;
    AllGPU_reduce allGPU_reduce;

    void restart(uint32_t restartStep, const std::string restartDirectory)
    {
        if(this->notifyPeriod.empty())
            return;

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
            filename << restartDirectory << "/" << prefix << "_" << restartStep;

            hdf5DataFile.open(filename.str().c_str(), fAttr);

            splash::Dimensions dimensions;

            hdf5DataFile.read(restartStep,
                              this->leftParticlesDatasetName.c_str(),
                              dimensions,
                              &(*hBufLeftParsCalorimeter.origin()));

            hdf5DataFile.close();

            /* rank 0 divides and distributes the calorimeter to all ranks in equal parts */
            uint32_t numRanks = gridCon.getGlobalSize();
            // get a host accelerator
            auto hostDev = cupla::manager::Device< cupla::AccHost >::get().device( );
            pmacc::algorithm::host::Foreach()(hostDev,
                                              hBufLeftParsCalorimeter.zone(),
                                              hBufLeftParsCalorimeter.origin(),
                                              DivideInPlace<float_X>(float_X(numRanks)));
        }

        MPI_Bcast(&(*hBufLeftParsCalorimeter.origin()),
                  hBufLeftParsCalorimeter.size().productOfComponents() * sizeof(float_X),
                  MPI_CHAR,
                  0, /* rank 0 */
                  comm.getMPIComm());

        *this->dBufLeftParsCalorimeter = hBufLeftParsCalorimeter;
    }


    void checkpoint(uint32_t currentStep, const std::string checkpointDirectory)
    {
        if(this->notifyPeriod.empty())
            return;

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
        filename << checkpointDirectory << "/" << prefix << "_" << currentStep;

        hdf5DataFile.open(filename.str().c_str(), fAttr);

        typename PICToSplash<float_X>::type SplashTypeX;

        splash::Dimensions bufferSize(hBufTotal.size().x(),
                                      hBufTotal.size().y(),
                                      hBufTotal.size().z());

        /* if there is only one energy bin, omit the energy axis */
        uint32_t dimension = this->numBinsEnergy == 1 ? DIM2 : DIM3;
        hdf5DataFile.write(currentStep,
                           SplashTypeX,
                           dimension,
                           splash::Selection(bufferSize),
                           this->leftParticlesDatasetName.c_str(),
                           &(*hBufTotal.origin()));

        hdf5DataFile.close();
    }


    void pluginLoad()
    {
        namespace pm = pmacc::math;

        if(this->notifyPeriod.empty())
            return;

        if(!(this->openingYaw_deg > float_X(0.0) && this->openingYaw_deg <= float_X(360.0)))
        {
            std::stringstream msg;
            msg << "[Plugin] [" << this->prefix
                << "] openingYaw has to be within (0, 360]."
                << std::endl;
            throw std::runtime_error(msg.str());
        }
        if(!(this->openingPitch_deg > float_X(0.0) && this->openingPitch_deg <= float_X(180.0)))
        {
            std::stringstream msg;
            msg << "[Plugin] [" << this->prefix
                << "] openingPitch has to be within (0, 180]."
                << std::endl;
            throw std::runtime_error(msg.str());
        }
        if(this->numBinsEnergy > 1 && this->maxEnergy <= this->minEnergy)
        {
            std::stringstream msg;
            msg << "[Plugin] [" << this->prefix
                << "] minEnergy has to be less than maxEnergy."
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
        pm::Size_t<simDim> gpuDim = (pm::Size_t<simDim>)con.getGpuNodes();
        zone::SphericZone<simDim> zone_allGPUs(gpuDim);
        this->allGPU_reduce = AllGPU_reduce(new pmacc::algorithm::mpi::Reduce<simDim>(zone_allGPUs));

        /* calculate rotated calorimeter frame from posYaw_deg and posPitch_deg */
        const float_64 posYaw_rad = this->posYaw_deg * float_64(M_PI / 180.0);
        const float_64 posPitch_rad = this->posPitch_deg * float_64(M_PI / 180.0);
        this->calorimeterFrameVecY = float3_X(math::sin(posYaw_rad) * math::cos(posPitch_rad),
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
            this->calorimeterFrameVecX = math::cross(vecUp, this->calorimeterFrameVecY);
            /* normalize vector */
            this->calorimeterFrameVecX /= math::abs(this->calorimeterFrameVecX);
        }
        this->calorimeterFrameVecZ = math::cross(this->calorimeterFrameVecX, this->calorimeterFrameVecY);

        /* create calorimeter functor instance */
        this->calorimeterFunctor = MyCalorimeterFunctorPtr(new MyCalorimeterFunctor(
            this->maxYaw_deg * float_X(M_PI / 180.0),
            this->maxPitch_deg * float_X(M_PI / 180.0),
            this->numBinsYaw,
            this->numBinsPitch,
            this->numBinsEnergy,
            this->logScale ? math::log10(this->minEnergy) : this->minEnergy,
            this->logScale ? math::log10(this->maxEnergy) : this->maxEnergy,
            this->logScale,
            this->calorimeterFrameVecX,
            this->calorimeterFrameVecY,
            this->calorimeterFrameVecZ));

        /* create folder for hdf5 files*/
        Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(this->foldername);

        Environment<>::get().PluginConnector().setNotificationPeriod(this, this->notifyPeriod);
    }


    void pluginUnload()
    {
        if(this->notifyPeriod.empty())
            return;

        __delete(this->dBufCalorimeter);
        __delete(this->dBufLeftParsCalorimeter);
        __delete(this->hBufCalorimeter);
        __delete(this->hBufTotalCalorimeter);
    }


    void writeToHDF5File(uint32_t currentStep)
    {
        splash::SerialDataCollector hdf5DataFile(1);
        splash::DataCollector::FileCreationAttr fAttr;

        splash::DataCollector::initFileCreationAttr(fAttr);

        std::stringstream filename;
        filename << this->foldername << "/" << this->prefix << "_" << currentStep;

        hdf5DataFile.open(filename.str().c_str(), fAttr);

        typename PICToSplash<float_X>::type SplashTypeX;
        typename PICToSplash<float_64>::type SplashType64;
        typename PICToSplash<bool>::type SplashTypeBool;

        splash::Dimensions bufferSize(this->hBufTotalCalorimeter->size().x(),
                                      this->hBufTotalCalorimeter->size().y(),
                                      this->hBufTotalCalorimeter->size().z());

        hdf5DataFile.write(currentStep,
                           SplashTypeX,
                           this->numBinsEnergy == 1 ? DIM2 : DIM3,
                           splash::Selection(bufferSize),
                           "calorimeter",
                           &(*this->hBufTotalCalorimeter->origin()));

        const float_64 unitSI = particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE * UNIT_ENERGY;

        hdf5DataFile.writeAttribute(currentStep,
                                    SplashType64,
                                    "calorimeter",
                                    "unitSI",
                                    &unitSI);

        hdf5DataFile.writeAttribute(currentStep,
                                    SplashType64,
                                    "calorimeter",
                                    "posYaw[deg]",
                                    &posYaw_deg);

        hdf5DataFile.writeAttribute(currentStep,
                                    SplashType64,
                                    "calorimeter",
                                    "posPitch[deg]",
                                    &posPitch_deg);

        hdf5DataFile.writeAttribute(currentStep,
                                    SplashTypeX,
                                    "calorimeter",
                                    "maxYaw[deg]",
                                    &this->maxYaw_deg);

        hdf5DataFile.writeAttribute(currentStep,
                                    SplashTypeX,
                                    "calorimeter",
                                    "maxPitch[deg]",
                                    &this->maxPitch_deg);

        if(this->numBinsEnergy > 1)
        {
            const float_64 minEnergy_SI = this->minEnergy * UNIT_ENERGY;
            const float_64 maxEnergy_SI = this->maxEnergy * UNIT_ENERGY;
            const float_64 minEnergy_keV = minEnergy_SI * UNITCONV_Joule_to_keV;
            const float_64 maxEnergy_keV = maxEnergy_SI * UNITCONV_Joule_to_keV;

            hdf5DataFile.writeAttribute(currentStep,
                                        SplashType64,
                                        "calorimeter",
                                        "minEnergy[keV]",
                                        &minEnergy_keV);

            hdf5DataFile.writeAttribute(currentStep,
                                        SplashType64,
                                        "calorimeter",
                                        "maxEnergy[keV]",
                                        &maxEnergy_keV);

            hdf5DataFile.writeAttribute(currentStep,
                                        SplashTypeBool,
                                        "calorimeter",
                                        "logScale",
                                        &this->logScale);
        }

        hdf5DataFile.close();
    }

public:
    ParticleCalorimeter() :
        name("ParticleCalorimeter: (virtually) propagates and collects particles to infinite distance"),
        prefix(ParticlesType::FrameType::getName() + std::string("_calorimeter")),
        foldername(prefix),
        cellDescription(nullptr),
        leftParticlesDatasetName("calorimeterLeftParticles"),
        dBufCalorimeter(nullptr),
        dBufLeftParsCalorimeter(nullptr),
        hBufCalorimeter(nullptr),
        hBufTotalCalorimeter(nullptr)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }


    void notify(uint32_t currentStep)
    {
        /* initialize calorimeter with already detected particles */
        *this->dBufCalorimeter = *this->dBufLeftParsCalorimeter;

        /* data is written to dBufCalorimeter */
        this->calorimeterFunctor->setCalorimeterCursor(this->dBufCalorimeter->origin());

        /* create kernel functor instance */
        DataConnector &dc = Environment<>::get().DataConnector();
        auto particles = dc.get< ParticlesType >( ParticlesType::FrameType::getName(), true );

        AreaMapping<
            CORE + BORDER,
            MappingDesc
        > const mapper( *this->cellDescription );
        auto const grid = mapper.getGridDim();

        constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
            pmacc::math::CT::volume< SuperCellSize >::type::value
        >::value;

        PMACC_KERNEL( KernelParticleCalorimeter< numWorkers >{ } )(
            grid,
            numWorkers
        )(
            particles->getDeviceParticlesBox( ),
            *this->calorimeterFunctor,
            mapper
        );

        dc.releaseData( ParticlesType::FrameType::getName() );

        /* copy to host */
        *this->hBufCalorimeter = *this->dBufCalorimeter;

        /* mpi reduce */
        (*this->allGPU_reduce)(*this->hBufTotalCalorimeter, *this->hBufCalorimeter, pmacc::algorithm::functor::Add{});
        if(!this->allGPU_reduce->root())
            return;

        this->writeToHDF5File(currentStep);
    }


    void setMappingDescription(MappingDesc* cellDescription)
    {
        this->cellDescription = cellDescription;
    }


    void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
        ((this->prefix + ".period").c_str(), po::value<std::string> (&this->notifyPeriod),
            "enable plugin [for each n-th step]")
        ((this->prefix + ".numBinsYaw").c_str(), po::value<uint32_t > (&this->numBinsYaw)->default_value(64),
            "number of bins for angle yaw.")
        ((this->prefix + ".numBinsPitch").c_str(), po::value<uint32_t > (&this->numBinsPitch)->default_value(64),
            "number of bins for angle pitch.")
        ((this->prefix + ".numBinsEnergy").c_str(), po::value<uint32_t > (&this->numBinsEnergy)->default_value(1),
            "number of bins for the energy spectrum. Disabled by default.")
        ((this->prefix + ".minEnergy").c_str(), po::value<float_X > (&this->minEnergy)->default_value(float_X(0.0)),
            "minimal detectable energy in keV.")
        ((this->prefix + ".maxEnergy").c_str(), po::value<float_X > (&this->maxEnergy)->default_value(float_X(1.0e3)),
            "maximal detectable energy in keV.")
        ((this->prefix + ".logScale").c_str(), po::bool_switch(&this->logScale),
            "enable logarithmic energy scale.")
        ((this->prefix + ".openingYaw").c_str(), po::value<float_X > (&this->openingYaw_deg)->default_value(float_X(360.0)),
            "opening angle yaw in degrees. 0 < x < 360.")
        ((this->prefix + ".openingPitch").c_str(), po::value<float_X > (&this->openingPitch_deg)->default_value(float_X(180.0)),
            "opening angle pitch in degrees. 0 < x < 180.")
        ((this->prefix + ".posYaw").c_str(), po::value<float_64 > (&this->posYaw_deg)->default_value(float_64(0.0)),
            "yaw coordinate of calorimeter position in degrees. Defaults to +y direction.")
        ((this->prefix + ".posPitch").c_str(), po::value<float_64 > (&this->posPitch_deg)->default_value(float_64(0.0)),
            "pitch coordinate of calorimeter position in degrees. Defaults to +y direction.");
    }


    std::string pluginGetName() const
    {
        return this->name;
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

        DataConnector &dc = Environment<>::get().DataConnector();
        auto particles = dc.get< ParticlesType >( speciesName, true );

        constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
            pmacc::math::CT::volume< SuperCellSize >::type::value
        >::value;

        PMACC_KERNEL( KernelParticleCalorimeter< numWorkers >{ } )(
            grid,
            numWorkers
        )(
            particles->getDeviceParticlesBox( ),
            (MyCalorimeterFunctor)*this->calorimeterFunctor,
            mapper
        );
        dc.releaseData( speciesName );
    }
};

namespace particles
{
namespace traits
{
    template<
        typename T_Species,
        typename T_UnspecifiedSpecies
    >
    struct SpeciesEligibleForSolver<
        T_Species,
        ParticleCalorimeter< T_UnspecifiedSpecies >
    >
    {
        using FrameType = typename T_Species::FrameType;

        // this plugin needs at least the weighting and momentum attributes
        using RequiredIdentifiers = MakeSeq_t<
            weighting,
            momentum
        >;

        using SpeciesHasIdentifiers = typename pmacc::traits::HasIdentifiers<
            FrameType,
            RequiredIdentifiers
        >::type;

        // and also a mass ratio for energy calculation from momentum
        using SpeciesHasFlags = typename pmacc::traits::HasFlag<
            FrameType,
            massRatio<>
        >::type;

        using type = typename bmpl::and_<
            SpeciesHasIdentifiers,
            SpeciesHasFlags
        >;
    };
} // namespace traits
} // namespace particles
} // namespace picongpu
