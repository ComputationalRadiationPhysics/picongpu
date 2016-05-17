/**
 * Copyright 2016 Heiko Burau
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

#include "traits/PICToSplash.hpp"
#include "plugins/ISimulationPlugin.hpp"
#include "cuSTL/container/DeviceBuffer.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/algorithm/kernel/Foreach.hpp"
#include "cuSTL/cursor/MultiIndexCursor.hpp"
#include "cuSTL/algorithm/mpi/Reduce.hpp"
#include "cuSTL/algorithm/host/Foreach.hpp"
#include "particles/policies/ExchangeParticles.hpp"
#include "math/Vector.hpp"
#include "algorithms/math.hpp"
#include <boost/shared_ptr.hpp>

/* libSplash data output */
#include <splash/splash.h>
#include <boost/filesystem.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>

namespace picongpu
{
using namespace PMacc;

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
    uint32_t notifyPeriod;
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

    typedef PMacc::container::DeviceBuffer<float_X, DIM3> DBufCalorimeter;
    typedef PMacc::container::HostBuffer<float_X, DIM3> HBufCalorimeter;

    /* device calorimeter buffer for a single gpu */
    DBufCalorimeter* dBufCalorimeter;
    /* device calorimeter buffer for all particles which have left the simulation volume  */
    DBufCalorimeter* dBufLeftParsCalorimeter;
    /* host calorimeter buffer for a single mpi rank */
    HBufCalorimeter* hBufCalorimeter;
    /* host calorimeter buffer for summation of all mpi ranks */
    HBufCalorimeter* hBufTotalCalorimeter;

public:
    typedef CalorimeterFunctor<typename DBufCalorimeter::Cursor> MyCalorimeterFunctor;
private:
    typedef boost::shared_ptr<MyCalorimeterFunctor> MyCalorimeterFunctorPtr;
    MyCalorimeterFunctorPtr calorimeterFunctor;

    typedef boost::shared_ptr<PMacc::algorithm::mpi::Reduce<simDim> > AllGPU_reduce;
    AllGPU_reduce allGPU_reduce;

    void restart(uint32_t restartStep, const std::string restartDirectory)
    {
        if(this->notifyPeriod == 0)
            return;

        HBufCalorimeter hBufLeftParsCalorimeter(this->dBufLeftParsCalorimeter->size());

        PMacc::GridController<simDim>& gridCon = PMacc::Environment<simDim>::get().GridController();
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
            using namespace lambda;
            PMacc::algorithm::host::Foreach()(hBufLeftParsCalorimeter.zone(),
                                              hBufLeftParsCalorimeter.origin(),
                                              _1 = _1 / float_X(numRanks));
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
        if(this->notifyPeriod == 0)
            return;

        HBufCalorimeter hBufLeftParsCalorimeter(this->dBufLeftParsCalorimeter->size());
        HBufCalorimeter hBufTotal(hBufLeftParsCalorimeter.size());

        hBufLeftParsCalorimeter = *this->dBufLeftParsCalorimeter;

        /* mpi reduce */
        using namespace lambda;
        (*this->allGPU_reduce)(hBufTotal, hBufLeftParsCalorimeter, _1 + _2);
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
        namespace pm = PMacc::math;
        namespace pam = PMacc::algorithms::math;

        if(this->notifyPeriod == 0)
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
        PMacc::GridController<simDim>& con = PMacc::Environment<simDim>::get().GridController();
        pm::Size_t<simDim> gpuDim = (pm::Size_t<simDim>)con.getGpuNodes();
        zone::SphericZone<simDim> zone_allGPUs(gpuDim);
        this->allGPU_reduce = AllGPU_reduce(new PMacc::algorithm::mpi::Reduce<simDim>(zone_allGPUs));

        /* calculate rotated calorimeter frame from posYaw_deg and posPitch_deg */
        const float_64 posYaw_rad = this->posYaw_deg * float_64(M_PI / 180.0);
        const float_64 posPitch_rad = this->posPitch_deg * float_64(M_PI / 180.0);
        this->calorimeterFrameVecY = float3_X(pam::sin(posYaw_rad) * pam::cos(posPitch_rad),
                                              pam::cos(posYaw_rad) * pam::cos(posPitch_rad),
                                              pam::sin(posPitch_rad));
        /* If the y-axis is pointing exactly up- or downwards we need to define the x-axis manually */
        if(pam::abs(this->calorimeterFrameVecY.z()) == float_X(1.0))
        {
            this->calorimeterFrameVecX = float3_X(1.0, 0.0, 0.0);
        }
        else
        {
            /* choose `calorimeterFrameVecX` so that the roll is zero. */
            const float3_X vecUp(0.0, 0.0, -1.0);
            this->calorimeterFrameVecX = pam::cross(vecUp, this->calorimeterFrameVecY);
            /* normalize vector */
            this->calorimeterFrameVecX /= pam::abs(this->calorimeterFrameVecX);
        }
        this->calorimeterFrameVecZ = pam::cross(this->calorimeterFrameVecX, this->calorimeterFrameVecY);

        /* create calorimeter functor instance */
        this->calorimeterFunctor = MyCalorimeterFunctorPtr(new MyCalorimeterFunctor(
            this->maxYaw_deg * float_X(M_PI / 180.0),
            this->maxPitch_deg * float_X(M_PI / 180.0),
            this->numBinsYaw,
            this->numBinsPitch,
            this->numBinsEnergy,
            this->logScale ? pam::log10(this->minEnergy) : this->minEnergy,
            this->logScale ? pam::log10(this->maxEnergy) : this->maxEnergy,
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
        if(this->notifyPeriod == 0)
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
        notifyPeriod(0),
        cellDescription(NULL),
        leftParticlesDatasetName("calorimeterLeftParticles"),
        dBufCalorimeter(NULL),
        dBufLeftParsCalorimeter(NULL),
        hBufCalorimeter(NULL),
        hBufTotalCalorimeter(NULL)
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
        ParticlesType* particles = &(dc.getData<ParticlesType > (ParticlesType::FrameType::getName(), true));

        ParticleCalorimeterKernel<typename ParticlesType::ParticlesBoxType,
                                  MyCalorimeterFunctor>
                                  particleCalorimeterKernel(particles->getDeviceParticlesBox(),
                                                            *this->calorimeterFunctor);

        /* create zone */
        typedef typename MappingDesc::SuperCellSize SuperCellSize;

        const PMacc::math::Int<simDim> coreBorderGuardSuperCells = this->cellDescription->getGridSuperCells();
        const uint32_t guardSuperCells = this->cellDescription->getGuardingSuperCells();
        const PMacc::math::Int<simDim> coreBorderSuperCells = coreBorderGuardSuperCells - 2*guardSuperCells;

        /* this zone represents the core+border area with guard offset in unit of cells */
        const zone::SphericZone<simDim> zone(
            static_cast<PMacc::math::Size_t<simDim> >(coreBorderSuperCells * SuperCellSize::toRT()),
            guardSuperCells * SuperCellSize::toRT());

        /* kernel call */
        algorithm::kernel::Foreach<SuperCellSize> foreach;
        foreach(zone, cursor::make_MultiIndexCursor<simDim>(), particleCalorimeterKernel);

        /* copy to host */
        *this->hBufCalorimeter = *this->dBufCalorimeter;

        /* mpi reduce */
        using namespace lambda;
        (*this->allGPU_reduce)(*this->hBufTotalCalorimeter, *this->hBufCalorimeter, _1 + _2);
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
        ((this->prefix + ".period").c_str(), po::value<uint32_t > (&this->notifyPeriod)->default_value(0),
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
        if(this->notifyPeriod == 0)
            return;
        if(speciesName != ParticlesType::FrameType::getName())
            return;

        /* data is written to dBufLeftParsCalorimeter */
        this->calorimeterFunctor->setCalorimeterCursor(this->dBufLeftParsCalorimeter->origin());

        ExchangeMapping<GUARD, MappingDesc> mapper(*this->cellDescription, direction);
        dim3 grid(mapper.getGridDim());

        DataConnector &dc = Environment<>::get().DataConnector();
        ParticlesType* particles = &(dc.getData<ParticlesType > (speciesName, true));

        __cudaKernel(kernelParticleCalorimeter)
                (grid, mapper.getSuperCellSize())
                (particles->getDeviceParticlesBox(), (MyCalorimeterFunctor)*this->calorimeterFunctor, mapper);
    }
};

} // namespace picongpu
