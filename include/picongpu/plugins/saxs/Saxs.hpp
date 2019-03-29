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

#include "picongpu/plugins/saxs/Saxs.kernel"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/common/stringHelpers.hpp"

#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/dimensions/DataSpaceOperations.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/traits/HasIdentifier.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>

#include <boost/filesystem.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>


namespace picongpu
{
using namespace pmacc;

namespace po = boost::program_options;


///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////  SAXS Plugin Class  ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template<class ParticlesType>
class Saxs : public ISimulationPlugin
{
private:

    typedef MappingDesc::SuperCellSize SuperCellSize;

    GridBuffer<float1_64, DIM1> *sumfcoskr;
    GridBuffer<float1_64, DIM1> *sumfsinkr;
    GridBuffer<float1_X, DIM1> *np;
    // GridBuffer<int64_t, DIM1> *nmp;

    MappingDesc *cellDescription;
    std::string notifyPeriod;
    std::string speciesName;
    std::string pluginName;
    std::string pluginPrefix;
    std::string filename_prefix;
    float3_64 q_min, q_max, q_step;
    unsigned int n_qx, n_qy, n_qz, n_q;
    float1_64 *sumfcoskr_master;
    float1_64 *sumfsinkr_master;
    float1_64 *intensity_master;
    float1_X np_master; // Number of particles
    // int64_t nmp_master; // Number of macro particles

    bool isMaster;

    uint32_t currentStep;

    mpi::MPIReduce reduce;

public:

    Saxs() :
    pluginName("SAXS: calculate the SAXS scattering intensity of a species"),
    speciesName(ParticlesType::FrameType::getName()),
    pluginPrefix(speciesName + std::string("_saxs")),
    filename_prefix(pluginPrefix),
    sumfcoskr(nullptr),
    sumfsinkr(nullptr),
    np(nullptr),
    // nmp(nullptr),
    cellDescription(nullptr),
    isMaster(false),
    currentStep(0)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }   
   

    virtual ~Saxs()
    {
    }

    /**
     * This function represents what is actually calculated if the plugin
     * is called. Here, one only sets the particles pointer to the data of
     * the latest time step and calls the 'calculateRadiationParticles'
     * function if for the actual time step radiation is to be calculated.
     * @param currentStep
     */
    void notify(uint32_t currentStep)
    {
        /* CORE + BORDER is PIC black magic, currently not needed
            *
            */
        std::cout << "SAXS plugin enabled" << std::endl;
        calculateSAXS < CORE + BORDER > (currentStep);
    }

    void pluginRegisterHelp(po::options_description& desc)
    {
            desc.add_options()
                ((pluginPrefix + ".period").c_str(), po::value<std::string> (&notifyPeriod), "enable plugin [for each n-th step]")
                ((pluginPrefix + ".qx_max").c_str(), po::value<float_64 > (&q_max[0])->default_value(5), "reciprocal space range qx_max (A^-1)")
                ((pluginPrefix + ".qy_max").c_str(), po::value<float_64 > (&q_max[1])->default_value(5), "reciprocal space range qy_max (A^-1)")
                ((pluginPrefix + ".qz_max").c_str(), po::value<float_64 > (&q_max[2])->default_value(5), "reciprocal space range qz_max (A^-1)")
                ((pluginPrefix + ".qx_min").c_str(), po::value<float_64 > (&q_min[0])->default_value(-5), "reciprocal space range qx_min (A^-1)")
                ((pluginPrefix + ".qy_min").c_str(), po::value<float_64 > (&q_min[1])->default_value(-5), "reciprocal space range qy_min (A^-1)")
                ((pluginPrefix + ".qz_min").c_str(), po::value<float_64 > (&q_min[2])->default_value(-5), "reciprocal space range qz_min (A^-1)")
                ((pluginPrefix + ".n_qx").c_str(), po::value<unsigned int> (&n_qx)->default_value(100), "Number of qx")
                ((pluginPrefix + ".n_qy").c_str(), po::value<unsigned int> (&n_qy)->default_value(100), "Number of qy")
                ((pluginPrefix + ".n_qz").c_str(), po::value<unsigned int> (&n_qz)->default_value(1), "Number of qz");

    }


    std::string pluginGetName() const
    {
        return pluginName;
    }


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
    
// static unsigned int getNq(unsigned int n_qx,unsigned int n_qy,unsigned int n_qz)
// {
//     unsigned int n_q = n_qx * n_qy * n_qz;
//     return n_q;
// }
    // template<typename gb>
    // void createQ(gb& q, float_X q_min, float_X q_step, unsigned int n_q)
    // {   
    //     for (int i = 0; i < n_q; i++)
    //         q[i] = q_min + q_step * i;
    // }


    /**
     * The plugin is loaded on every MPI rank, and therefor this function is
     * executed on every MPI rank.
     * One host with MPI rank 0 is defined to be the master.
     * It creates a folder where all the
     * results are saved and, depending on the type of radiation calculation,
     * creates an additional data structure for the summation of all
     * intermediate values.
     * On every host data structure for storage of the calculated radiation
     * is created.       */
    void pluginLoad()
    {
        if(!notifyPeriod.empty())
        {
            /*only rank 0 create a file*/
            // const unsigned int n_qx = getNq(qx_min,qx_max,qx_step);
            // const unsigned int n_qy = getNq(qy_min,qy_max,qy_step);
            // const unsigned int n_qz = getNq(qz_min,qz_max,qz_step);
            isMaster = reduce.hasResult(mpi::reduceMethods::Reduce());
            // qx = new gridBuffers<float_X,DIM1>(DataSpace<DIM1> (n_qx));
            // qy = new gridBuffers<float_X,DIM1>(DataSpace<DIM1> (n_qy));
            // qz = new gridBuffers<float_X,DIM1>(DataSpace<DIM1> (n_qz));
            // createQ(qx, qx_min, qx_step);
            // createQ(qy, qy_min, qy_step);
            // createQ(qz, qz_min, qz_step);
            n_q = n_qx * n_qy * n_qz;
            sumfcoskr = new GridBuffer<float1_64, DIM1 > (DataSpace<DIM1> (n_q)); //create one int on GPU and host
            sumfsinkr = new GridBuffer<float1_64, DIM1 > (DataSpace<DIM1> (n_q)); //create one int on GPU and host
            np = new GridBuffer<float1_X, DIM1 > (DataSpace<DIM1> (1)); //create one int on GPU and host
            // nmp = new GridBuffer<int64_t, DIM1 > (DataSpace<DIM1> (1)); //create one int on GPU and host
            sumfcoskr_master = new float1_64[n_q];
            sumfsinkr_master = new float1_64[n_q];
            intensity_master = new float1_64[n_q];
            Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
            pmacc::Filesystem<simDim>& fs = Environment<simDim>::get().Filesystem();

            if (isMaster)
            {
                fs.createDirectory("saxsOutput");
                fs.setDirectoryPermissions("saxsOutput");
            }
        }
    }


    void writeQintensity(float1_64* intensity, std::string name)
    {
        std::ofstream ofile;
        ofile.open(name.c_str(), std::ofstream::out | std::ostream::trunc);
        if (!ofile)
        {
            std::cerr << "Can't open file [" << name << "] for output, disable plugin output.\n";
            isMaster = false; // no Master anymore -> no process is able to write
        }
        else
        {
            int i_x, i_y, i_z;
            float3_64 q;

            ofile << n_q << "\n";
            ofile << "# qx qy qz intensity \n";
            for (unsigned int i = 0; i < n_q ; i++)
            {
                i_z = i % n_qz ;
                i_y = (i / n_qz) % n_qy ;
                i_x = i / (n_qz * n_qy) ;
                q[0] = q_min[0] + q_step[0] * i_x;
                q[1] = q_min[1] + q_step[1] * i_y;
                q[2] = q_min[2] + q_step[2] * i_z;
                ofile << q[0] << " " << q[1] << " " << q[2] << " " << intensity[i][0] << "\n";
            }
            ofile.flush();

            if (ofile.fail())
                std::cerr << "Error on flushing file [" << name << "]. " << std::endl;

            ofile.close();
        }
    }

    void writeLog(float1_X np_master,int64_t nmp_master, std::string name)
    {
        std::ofstream ofile;
        ofile.open(name.c_str(), std::ofstream::out | std::ostream::trunc);
        if (!ofile)
        {
            std::cerr << "Can't open file [" << name << "] for output, disable plugin output.\n";
            isMaster = false; // no Master anymore -> no process is able to write
        }
        else
        {
            ofile <<  "Number of particles:"  << " " << np_master.x() << "\n";
            ofile <<  "Number of macro particles:"  << " " << nmp_master << "\n";
            ofile.flush();

            if (ofile.fail())
                std::cerr << "Error on flushing file [" << name << "]. " << std::endl;

            ofile.close();
        }
    }


    void pluginUnload()
    {
        if(!notifyPeriod.empty())
        {
            __delete(sumfcoskr);
            __delete(sumfsinkr);
            __delete(np);
            // __delete(nmp);
            __deleteArray(intensity_master);
            CUDA_CHECK(cudaGetLastError());
        }
    }


  /** Method to copy data from GPU to CPU */
  void copyIntensityDeviceToHost()
  {
    sumfcoskr->deviceToHost();
    sumfsinkr->deviceToHost();
    np->deviceToHost();
    // nmp->deviceToHost();
    __getTransactionEvent().waitForFinished();
  }




  /** combine radiation data from each CPU and store result on master
   *  copyRadiationDeviceToHost() should be called before */
  void collectIntensityOnMaster()
  {
      reduce(nvidia::functors::Add(),
             sumfcoskr_master,
             sumfcoskr->getHostBuffer().getBasePointer(),
             n_q,
             mpi::reduceMethods::Reduce()
             );
      reduce(nvidia::functors::Add(),
             sumfsinkr_master,
             sumfsinkr->getHostBuffer().getBasePointer(),
             n_q,
             mpi::reduceMethods::Reduce()
             );
      reduce(nvidia::functors::Add(),
             &np_master,
             np->getHostBuffer().getBasePointer(),
             1,
             mpi::reduceMethods::Reduce()
             );
/*       reduce(nvidia::functors::add(),
             &nmp_master,
             nmp->gethostbuffer().getbasepointer(),
             1,
             mpi::reducemethods::reduce()
             ); */
        
      // Calculate intensity on master
      if (isMaster)
      {
        // printf("np_master = %f\n", np_master.x());
        for ( unsigned int i = 0; i < n_q; i++)
            intensity_master[i] = (sumfcoskr_master[i]*sumfcoskr_master[i]+sumfsinkr_master[i]*sumfsinkr_master[i])/np_master.x();
        std::stringstream o_step;
              o_step << currentStep;
        writeQintensity(intensity_master, "saxsOutput/" + filename_prefix + "_" + o_step.str() + ".dat");
        writeLog(np_master,nmp_master,"saxsOutput/" + filename_prefix + "_" + o_step.str() + ".log");
      } 
        
  }




  /** perform all operations to get data from GPU to master */
  void collectDataGPUToMaster()
  {
      // collect data GPU -> CPU -> Master
      copyIntensityDeviceToHost();
      collectIntensityOnMaster();
  }


 


  /**
   * This functions calls the radiation kernel. It specifies how the
   * calculation is parallelized.
   *      gridDim_rad is the number of Thread-Blocks in a grid
   *      blockDim_rad is the number of threads per block
   *
   * -----------------------------------------------------------
   * | Grid                                                    |
   * |   --------------   --------------                       |
   * |   |   Block 0  |   |   Block 1  |                       |
   * |   |o      o    |   |o      o    |                       |
   * |   |o      o    |   |o      o    |                       |
   * |   |th1    th2  |   |th1    th2  |                       |
   * |   --------------   --------------                       |
   * -----------------------------------------------------------
   *
   * !!! The TEMPLATE parameter is not used anymore.
   * !!! But the calculations it is supposed to do is hard coded in the
   *     kernel.
   * !!! THIS NEEDS TO BE CHANGED !!!
   *
   * @param currentStep
   */
  template< uint32_t AREA> /*This Template Parameter is not used anymore*/
  void calculateSAXS(uint32_t currentStep)
  {
      this->currentStep = currentStep;

      DataConnector &dc = Environment<>::get().DataConnector();
      auto particles = dc.get< ParticlesType >( ParticlesType::FrameType::getName(), true );



      /* number of threads per block = number of cells in a super cell
       *          = number of particles in a Frame
       *          (THIS IS PIConGPU SPECIFIC)
       * A Frame is the entity that stores particles.
       * A super cell can have many Frames.
       * Particles in a Frame can be accessed in parallel.
       */

      // Some funny things that make it possible for the kernel to calculate
      // the absolute position of the particles
      const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
      DataSpace<simDim> globalOffset(subGrid.getLocalDomain().offset);

      constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
          pmacc::math::CT::volume< SuperCellSize >::type::value
      >::value;

    // initialize variables with zero
    sumfcoskr->getDeviceBuffer( ).setValue( 0.0 );
    sumfsinkr->getDeviceBuffer( ).setValue( 0.0 );
    np->getDeviceBuffer( ).setValue( 0.0 );
    nmp->getDeviceBuffer( ).setValue( 0.0 );
    
    // calculate q_step
    q_step[0] = (q_max[0]-q_min[0])/n_qx;
    q_step[1] = (q_max[1]-q_min[1])/n_qy;
    q_step[2] = (q_max[2]-q_min[2])/n_qz;


    //   PIC-like kernel call of the saxs kernel
      PMACC_KERNEL( 
          KernelSaxs< numWorkers >{} 
      )(
            1,
            numWorkers
      )(
         /*Pointer to particles memory on the device*/
         particles->getDeviceParticlesBox(),
         /*Pointer to memory of sumfcoskr & sumfsinkr on the device*/
         sumfcoskr->getDeviceBuffer().getDataBox(),
         sumfsinkr->getDeviceBuffer().getDataBox(),
         np->getDeviceBuffer().getDataBox(),
        //  nmp->getDeviceBuffer().getDataBox(),
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

      dc.releaseData( ParticlesType::FrameType::getName() );

      collectDataGPUToMaster();
    //   writeAllFiles(globalOffset);
    
      // reset amplitudes on GPU back to zero
      sumfcoskr->getDeviceBuffer().reset(false);
      sumfsinkr->getDeviceBuffer().reset(false);
      np->getDeviceBuffer().reset(false);

  }

};

}


