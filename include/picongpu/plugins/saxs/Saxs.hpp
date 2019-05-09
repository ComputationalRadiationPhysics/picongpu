/* Copyright 2013-2018 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Klaus Steiniger, Felix Schmitt, Benjamin Worpitz,
 *                     Juncheng E
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

#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/common/stringHelpers.hpp"
#include "picongpu/plugins/saxs/Saxs.kernel"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/DataSpaceOperations.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>
#include <pmacc/traits/HasIdentifier.hpp>

#include <boost/filesystem.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace picongpu
{
using namespace pmacc;

namespace po = boost::program_options;

/** SAXS plugin
 * This SAXS plugin simulates the SAXS scattering pattern on-the-fly
 * from the particle positions obtained from PIConGPU.
 **/

template <class ParticlesType> class Saxs : public ISimulationPlugin
{
  private:
    using SuperCellSize = MappingDesc::SuperCellSize;

    //! The real part of structure factor
    GridBuffer<float1_64, DIM1> *sumfcoskr;
    //! The imaginary part of structure factor
    GridBuffer<float1_64, DIM1> *sumfsinkr;
    //! Number of real particles
    GridBuffer<float1_64, DIM1> *np;
    //! Number of macro particles
    GridBuffer<int64_t, DIM1> *nmp;

    MappingDesc *cellDescription;
    std::string notifyPeriod;
    std::string speciesName;
    std::string pluginName;
    std::string pluginPrefix;
    std::string filename_prefix;

    /** Range of scattering vector
     * The scattering vecotor here is defined as
     * 4*pi*sin(theta)/lambda, where 2theta is the angle
     * between scattered and incident beam
     **/
    float3_64 q_min, q_max, q_step;
    //! Number of scattering vectors
    unsigned int n_qx, n_qy, n_qz, n_q;

    float1_64 *sumfcoskr_master;
    float1_64 *sumfsinkr_master;
    float1_64 *intensity_master;
    float1_64 np_master;
    int64_t nmp_master;

    bool isMaster;

    uint32_t currentStep;

    mpi::MPIReduce reduce;

  public:
    Saxs()
        : pluginName(
              "SAXS: calculate the SAXS scattering intensity of a species"),
          speciesName(ParticlesType::FrameType::getName()),
          pluginPrefix(speciesName + std::string("_saxs")),
          filename_prefix(pluginPrefix), sumfcoskr(nullptr), sumfsinkr(nullptr),
          np(nullptr), nmp(nullptr), cellDescription(nullptr), isMaster(false),
          currentStep(0)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~Saxs() {}

    /**
     * This function represents what is actually calculated if the plugin
     * is called. Here, one only sets the particles pointer to the data of
     * the latest time step and calls the 'calculateSAXS'
     * function if for the actual time step radiation is to be calculated.
     * @param currentStep
     */
    void notify(uint32_t currentStep)
    {
        std::cout << "SAXS plugin enabled" << std::endl;
        calculateSAXS(currentStep);
    }

    void pluginRegisterHelp(po::options_description &desc)
    {
        desc.add_options()((pluginPrefix + ".period").c_str(),
            po::value<std::string>(&notifyPeriod),
            "enable plugin [for each n-th step]")(
            (pluginPrefix + ".qx_max").c_str(),
            po::value<float_64>(&q_max[0])->default_value(5),
            "reciprocal space range qx_max (A^-1)")(
            (pluginPrefix + ".qy_max").c_str(),
            po::value<float_64>(&q_max[1])->default_value(5),
            "reciprocal space range qy_max (A^-1)")(
            (pluginPrefix + ".qz_max").c_str(),
            po::value<float_64>(&q_max[2])->default_value(5),
            "reciprocal space range qz_max (A^-1)")(
            (pluginPrefix + ".qx_min").c_str(),
            po::value<float_64>(&q_min[0])->default_value(-5),
            "reciprocal space range qx_min (A^-1)")(
            (pluginPrefix + ".qy_min").c_str(),
            po::value<float_64>(&q_min[1])->default_value(-5),
            "reciprocal space range qy_min (A^-1)")(
            (pluginPrefix + ".qz_min").c_str(),
            po::value<float_64>(&q_min[2])->default_value(-5),
            "reciprocal space range qz_min (A^-1)")(
            (pluginPrefix + ".n_qx").c_str(),
            po::value<unsigned int>(&n_qx)->default_value(100),
            "Number of qx")((pluginPrefix + ".n_qy").c_str(),
            po::value<unsigned int>(&n_qy)->default_value(100),
            "Number of qy")((pluginPrefix + ".n_qz").c_str(),
            po::value<unsigned int>(&n_qz)->default_value(1), "Number of qz");
    }

    std::string pluginGetName() const { return pluginName; }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

    void restart(uint32_t timeStep, const std::string restartDirectory)
    {
        // Keep this empty
    }

    void checkpoint(uint32_t timeStep, const std::string restartDirectory)
    {
        // Keep this empty
    }

  private:
    /**
     * The plugin is loaded on every MPI rank, and therefore this function is
     * executed on every MPI rank.
     * One host with MPI rank 0 is defined to be the master.
     * It creates a folder where all the
     * results are saved in a plain text format.
     **/
    void pluginLoad()
    {
        if (!notifyPeriod.empty())
        {
            isMaster = reduce.hasResult(mpi::reduceMethods::Reduce());
            n_q = n_qx * n_qy * n_qz;
            sumfcoskr = new GridBuffer<float1_64, DIM1>(DataSpace<DIM1>(n_q));
            sumfsinkr = new GridBuffer<float1_64, DIM1>(DataSpace<DIM1>(n_q));
            // allocate one float on GPU and host
            np = new GridBuffer<float1_64, DIM1>(DataSpace<DIM1>(1));
            // allocate one int on GPU and host
            nmp = new GridBuffer<int64_t, DIM1>(DataSpace<DIM1>(1));
            sumfcoskr_master = new float1_64[n_q];
            sumfsinkr_master = new float1_64[n_q];
            intensity_master = new float1_64[n_q];
            Environment<>::get().PluginConnector().setNotificationPeriod(
                this, notifyPeriod);
            pmacc::Filesystem<simDim> &fs =
                Environment<simDim>::get().Filesystem();

            // only rank 0 create a file
            if (isMaster)
            {
                fs.createDirectory("saxsOutput");
                fs.setDirectoryPermissions("saxsOutput");
            }
        }
    }

    /**
     * Write scattering intensity for each q
     * @param intensity
     * @param name The name of output file
     **/
    void writeIntensity(float1_64 *intensity, std::string name)
    {
        std::ofstream ofile;
        ofile.open(name.c_str(), std::ofstream::out | std::ostream::trunc);
        if (!ofile)
        {
            std::cerr << "Can't open file [" << name
                      << "] for output, disable plugin output.\n";
            isMaster =
                false; // no Master anymore -> no process is able to write
        }
        else
        {
            int i_x, i_y, i_z;
            float3_64 q;

            ofile << n_q << "\n";
            ofile << "# qx qy qz intensity \n";
            for (unsigned int i = 0; i < n_q; i++)
            {
                i_z = i % n_qz;
                i_y = (i / n_qz) % n_qy;
                i_x = i / (n_qz * n_qy);
                q[0] = q_min[0] + q_step[0] * i_x;
                q[1] = q_min[1] + q_step[1] * i_y;
                q[2] = q_min[2] + q_step[2] * i_z;
                ofile << q[0] << " " << q[1] << " " << q[2] << " "
                      << intensity[i][0] << "\n";
            }
            ofile.close();
        }
    }

    /**
     * Write a log file for number of real particles and number of
     * macro particles.
     * @param np_master The number of real particles
     * @param nmp_master The number of macro particles
     * @param name The name of output file
     **/
    void writeLog(float1_64 np_master, int64_t nmp_master, std::string name)
    {
        std::ofstream ofile;
        ofile.open(name.c_str(), std::ofstream::out | std::ostream::trunc);
        if (!ofile)
        {
            std::cerr << "Can't open file [" << name
                      << "] for output, disable plugin output.\n";
            isMaster =
                false; // no Master anymore -> no process is able to write
        }
        else
        {
            ofile << "Number of particles:"
                  << " " << np_master.x() << "\n";
            ofile << "Number of macro particles:"
                  << " " << nmp_master << "\n";

            ofile.close();
        }
    }

    void pluginUnload()
    {
        if (!notifyPeriod.empty())
        {
            __delete(sumfcoskr);
            __delete(sumfsinkr);
            __delete(np);
            __delete(nmp);
            __deleteArray(intensity_master);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    //! Method to copy data from GPU to CPU
    void copyIntensityDeviceToHost()
    {
        sumfcoskr->deviceToHost();
        sumfsinkr->deviceToHost();
        np->deviceToHost();
        nmp->deviceToHost();
        __getTransactionEvent().waitForFinished();
    }

    /** Collect intensity data from each CPU and store result on master
     *  copyIntensityDeviceToHost should be called before */
    void collectIntensityOnMaster()
    {
        reduce(nvidia::functors::Add(), sumfcoskr_master,
            sumfcoskr->getHostBuffer().getBasePointer(), n_q,
            mpi::reduceMethods::Reduce());
        reduce(nvidia::functors::Add(), sumfsinkr_master,
            sumfsinkr->getHostBuffer().getBasePointer(), n_q,
            mpi::reduceMethods::Reduce());
        reduce(nvidia::functors::Add(), &np_master,
            np->getHostBuffer().getBasePointer(), 1,
            mpi::reduceMethods::Reduce());
        reduce(nvidia::functors::Add(), &nmp_master,
            nmp->getHostBuffer().getBasePointer(), 1,
            mpi::reduceMethods::Reduce());

        // Calculate intensity on master
        if (isMaster)
        {
            for (unsigned int i = 0; i < n_q; i++)
                intensity_master[i] =
                    (sumfcoskr_master[i] * sumfcoskr_master[i] +
                        sumfsinkr_master[i] * sumfsinkr_master[i]) /
                    np_master.x();

            std::stringstream o_step;
            o_step << currentStep;
            writeIntensity(intensity_master,
                "saxsOutput/" + filename_prefix + "_" + o_step.str() + ".dat");
            writeLog(np_master, nmp_master,
                "saxsOutput/" + filename_prefix + "_" + o_step.str() + ".log");
        }
    }

    // perform all operations to get data from GPU to master
    void collectDataGPUToMaster()
    {
        // collect data GPU -> CPU -> Master
        copyIntensityDeviceToHost();
        collectIntensityOnMaster();
    }

    /**
     * This functions calls the SAXS kernel. It specifies how the
     * calculation is parallelized.
     **/
    void calculateSAXS(uint32_t currentStep)
    {
        this->currentStep = currentStep;

        DataConnector &dc = Environment<>::get().DataConnector();
        auto particles =
            dc.get<ParticlesType>(ParticlesType::FrameType::getName(), true);

        // calculate the absolute position of the particles
        const SubGrid<simDim> &subGrid = Environment<simDim>::get().SubGrid();
        DataSpace<simDim> globalOffset(subGrid.getLocalDomain().offset);

        constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
            pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

        // initialize variables with zero
        sumfcoskr->getDeviceBuffer().setValue(0.0);
        sumfsinkr->getDeviceBuffer().setValue(0.0);
        np->getDeviceBuffer().setValue(0.0);
        nmp->getDeviceBuffer().setValue(0.0);

        // calculate q_step
        q_step[0] = (q_max[0] - q_min[0]) / n_qx;
        q_step[1] = (q_max[1] - q_min[1]) / n_qy;
        q_step[2] = (q_max[2] - q_min[2]) / n_qz;

        // PIC-like kernel call of the SAXS kernel
        PMACC_KERNEL(
            KernelSaxs< numWorkers >{}
        )(
            1,
            numWorkers
        )(
            // Pointer to particles memory on the device
            particles->getDeviceParticlesBox(),
            // Pointer to memory of sumfcoskr & sumfsinkr on the device
            sumfcoskr->getDeviceBuffer().getDataBox(),
            sumfsinkr->getDeviceBuffer().getDataBox(),
            np->getDeviceBuffer().getDataBox(),
            nmp->getDeviceBuffer().getDataBox(),
            globalOffset,
            currentStep,
            *cellDescription,
            subGrid.getGlobalDomain().size,
            q_min,
            q_max,
            q_step,
            n_qx,
            n_qy,
            n_qz,
            n_q
        );

        dc.releaseData(ParticlesType::FrameType::getName());

        collectDataGPUToMaster();

        // reset amplitudes on GPU back to zero
        sumfcoskr->getDeviceBuffer().reset(false);
        sumfsinkr->getDeviceBuffer().reset(false);
        np->getDeviceBuffer().reset(false);
        nmp->getDeviceBuffer().reset(false);
    }
};

} // namespace picongpu
