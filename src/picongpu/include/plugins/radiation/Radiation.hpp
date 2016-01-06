/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Klaus Steiniger, Felix Schmitt, Benjamin Worpitz
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

#if(ENABLE_HDF5 != 1)
#error The activated radiation plugin (radiationConfig.param) requires HDF5
#endif

#include "simulation_defines.hpp"

#include "traits/SplashToPIC.hpp"
#include "traits/PICToSplash.hpp"

#include "dimensions/DataSpaceOperations.hpp"

#include "simulation_classTypes.hpp"
#include "mappings/kernel/AreaMapping.hpp"
#include "plugins/ISimulationPlugin.hpp"
#include "plugins/common/stringHelpers.hpp"

#include "mpi/reduceMethods/Reduce.hpp"
#include "mpi/MPIReduce.hpp"
#include "nvidia/functors/Add.hpp"

#include "sys/stat.h"

#include "plugins/radiation/Radiation.kernel"

/* libSplash data output */
#include <splash/splash.h>
#include <boost/filesystem.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>

namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;



///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////  Radiation Plugin Class  ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template<class ParticlesType>
class Radiation : public ISimulationPlugin
{
private:

    typedef MappingDesc::SuperCellSize SuperCellSize;

    typedef PIConGPUVerboseRadiation radLog;

    /**
     * At the moment the ParticlesType is PIC_ELECTRONS
     * (This special class which stores information about the momentum
     * of the last two time steps)
     */
    ParticlesType *particles;

    /**
     * Object that stores the complex radiated amplitude on host and device.
     * Radiated amplitude is a function of theta (looking direction) and
     * frequency. Layout of the radiation array is:
     * [omega_1(theta_1),omega_2(theta_1),...,omega_N-omega(theta_1),
     *   omega_1(theta_2),omega_2(theta_2),...,omega_N-omega(theta_N-theta)]
     */
    GridBuffer<Amplitude, DIM1> *radiation;
    radiation_frequencies::InitFreqFunctor freqInit;
    radiation_frequencies::FreqFunctor freqFkt;

    MappingDesc *cellDescription;
    uint32_t notifyFrequency;
    uint32_t dumpPeriod;
    uint32_t radStart;
    uint32_t radEnd;

    std::string pluginName;
    std::string pluginPrefix;
    std::string filename_prefix;
    bool totalRad;
    bool lastRad;
    std::string folderLastRad;
    std::string folderTotalRad;
    std::string pathOmegaList;
    bool radPerGPU;
    std::string folderRadPerGPU;
    DataSpace<simDim> lastGPUpos;

    /**
     * Data structure for storage and summation of the intermediate values of
     * the calculated Amplitude from every host for every direction and
     * frequency.
     */
    Amplitude* timeSumArray;
    Amplitude *tmp_result;
    vector_64* detectorPositions;
    float_64* detectorFrequencies;

    bool isMaster;

    uint32_t currentStep;
    uint32_t lastStep;

    std::string pathRestart;
    std::string meshesPathName;
    std::string particlesPathName;

    mpi::MPIReduce reduce;
    bool compressionOn;

public:

    Radiation() :
    pluginName("Radiation: calculate the radiation of a species"),
    pluginPrefix(ParticlesType::FrameType::getName() + std::string("_radiation")),
    filename_prefix(pluginPrefix),
    particles(NULL),
    radiation(NULL),
    cellDescription(NULL),
    notifyFrequency(0),
    dumpPeriod(0),
    totalRad(false),
    lastRad(false),
    timeSumArray(NULL),
    tmp_result(NULL),
    detectorPositions(NULL),
    detectorFrequencies(NULL),
    isMaster(false),
    currentStep(0),
    radPerGPU(false),
    lastStep(0),
    meshesPathName("DetectorMesh/"),
    particlesPathName("DetectorParticle/"),
    compressionOn(false)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~Radiation()
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

        DataConnector &dc = Environment<>::get().DataConnector();

        particles = &(dc.getData<ParticlesType > (ParticlesType::FrameType::getName(), true));

        if (currentStep >= radStart)
        {
            // radEnd = 0 is default, calculates radiation until simulation
            // end
            if (currentStep <= radEnd || radEnd == 0)
            {
                log<radLog::SIMULATION_STATE > ("Radiation: calculate time step %1% ") % currentStep;

                /* CORE + BORDER is PIC black magic, currently not needed
                 *
                 */
                calculateRadiationParticles < CORE + BORDER > (currentStep);

                log<radLog::SIMULATION_STATE > ("Radiation: finished time step %1% ") % currentStep;
            }
        }
    }

    void pluginRegisterHelp(po::options_description& desc)
    {

        desc.add_options()
            ((pluginPrefix + ".period").c_str(), po::value<uint32_t > (&notifyFrequency), "enable plugin [for each n-th step]")
            ((pluginPrefix + ".dump").c_str(), po::value<uint32_t > (&dumpPeriod)->default_value(0), "dump integrated radiation from last dumped step [for each n-th step] (0 = only print data at end of simulation)")
            ((pluginPrefix + ".lastRadiation").c_str(), po::bool_switch(&lastRad), "enable calculation of integrated radiation from last dumped step")
            ((pluginPrefix + ".folderLastRad").c_str(), po::value<std::string > (&folderLastRad)->default_value("lastRad"), "folder in which the integrated radiation from last dumped step is written")
            ((pluginPrefix + ".totalRadiation").c_str(), po::bool_switch(&totalRad), "enable calculation of integrated radiation from start of simulation")
            ((pluginPrefix + ".folderTotalRad").c_str(), po::value<std::string > (&folderTotalRad)->default_value("totalRad"), "folder in which the integrated radiation from start of simulation is written")
            ((pluginPrefix + ".start").c_str(), po::value<uint32_t > (&radStart)->default_value(2), "time index when radiation should start with calculation")
            ((pluginPrefix + ".end").c_str(), po::value<uint32_t > (&radEnd)->default_value(0), "time index when radiation should end with calculation")
            ((pluginPrefix + ".omegaList").c_str(), po::value<std::string > (&pathOmegaList)->default_value("_noPath_"), "path to file containing all frequencies to calculate")
            ((pluginPrefix + ".radPerGPU").c_str(), po::bool_switch(&radPerGPU), "enable radiation output from each GPU individually")
            ((pluginPrefix + ".folderRadPerGPU").c_str(), po::value<std::string > (&folderRadPerGPU)->default_value("radPerGPU"), "folder in which the radiation of each GPU is written")
            ((pluginPrefix + ".compression").c_str(), po::bool_switch(&compressionOn), "enable compression of hdf5 output");
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
        // only load backup if radiation is calculated:
        if(notifyFrequency == 0)
            return;

        if(isMaster)
        {
            // this will lead to wrong lastRad output right after the checkpoint if the restart point is
            // not a dump point. The correct lastRad data can be reconstructed from hdf5 data
            // since text based lastRad output will be obsolete soon, this is not a problem
            readHDF5file(timeSumArray, restartDirectory + "/" + std::string("radRestart_"), timeStep);
            log<radLog::SIMULATION_STATE > ("Radiation: restart finished");
        }
    }


    void checkpoint(uint32_t timeStep, const std::string restartDirectory)
    {
        // only write backup if radiation is calculated:
        if(notifyFrequency == 0)
            return;

        // collect data GPU -> CPU -> Master
        copyRadiationDeviceToHost();
        collectRadiationOnMaster();
        sumAmplitudesOverTime(tmp_result, timeSumArray);

        // write backup file
        if (isMaster)
        {
            writeHDF5file(tmp_result, restartDirectory + "/" + std::string("radRestart_"));
        }
    }


private:

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
        // allocate memory for all amplitudes for temporal data collection
        tmp_result = new Amplitude[elements_amplitude()];

        if (notifyFrequency > 0)
        {
            /*only rank 0 create a file*/
            isMaster = reduce.hasResult(mpi::reduceMethods::Reduce());

            radiation = new GridBuffer<Amplitude, DIM1 > (DataSpace<DIM1 > (elements_amplitude())); //create one int on GPU and host

            freqInit.Init(pathOmegaList);
            freqFkt = freqInit.getFunctor();


            Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyFrequency);
            PMacc::Filesystem<simDim>& fs = Environment<simDim>::get().Filesystem();

            if (isMaster)
            {
                timeSumArray = new Amplitude[elements_amplitude()];

                /* save detector position / observation direction */
                detectorPositions = new vector_64[parameters::N_observer];
                for(uint32_t detectorIndex=0; detectorIndex < parameters::N_observer; ++detectorIndex)
                {
                    detectorPositions[detectorIndex] = radiation_observer::observation_direction(detectorIndex);
                }

                /* save detector frequencies */
                detectorFrequencies = new float_64[radiation_frequencies::N_omega];
                for(uint32_t detectorIndex=0; detectorIndex < radiation_frequencies::N_omega; ++detectorIndex)
                {
                    detectorFrequencies[detectorIndex] = freqFkt(detectorIndex);
                }

            }

            if (isMaster && totalRad)
            {
                fs.createDirectory("radiationHDF5");
                fs.setDirectoryPermissions("radiationHDF5");
            }


            if (isMaster && radPerGPU)
            {
                fs.createDirectory(folderRadPerGPU);
                fs.setDirectoryPermissions(folderRadPerGPU);
            }

            if (isMaster && totalRad)
            {
                //create folder for total output
                fs.createDirectory(folderTotalRad);
                fs.setDirectoryPermissions(folderTotalRad);
                for (unsigned int i = 0; i < elements_amplitude(); ++i)
                    timeSumArray[i] = Amplitude::zero();
            }
            if (isMaster && lastRad)
            {
                //create folder for total output
                fs.createDirectory(folderLastRad);
                fs.setDirectoryPermissions(folderLastRad);
            }

        }
    }


    void pluginUnload()
    {
        if (notifyFrequency > 0)
        {

            // Some funny things that make it possible for the kernel to calculate
            // the absolute position of the particles
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            DataSpace<simDim> localSize(subGrid.getLocalDomain().size);
            const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
            DataSpace<simDim> globalOffset(subGrid.getLocalDomain().offset);
            globalOffset.y() += (localSize.y() * numSlides);

            // only print data at end of simulation if no dump period was set
            if (dumpPeriod == 0)
            {
                collectDataGPUToMaster();
                writeAllFiles(globalOffset);
            }

            if (isMaster)
            {
                __deleteArray(timeSumArray);
                delete[] detectorPositions;
                delete[] detectorFrequencies;
            }

            __delete(radiation);
            CUDA_CHECK(cudaGetLastError());
        }

        __deleteArray(tmp_result);
    }


  /** Method to copy data from GPU to CPU */
  void copyRadiationDeviceToHost()
  {
    radiation->deviceToHost();
    __getTransactionEvent().waitForFinished();
  }


  /** write radiation from each GPU to file individually
   *  requires call of copyRadiationDeviceToHost() before */
  void saveRadPerGPU(const DataSpace<simDim> currentGPUpos)
  {
    if (radPerGPU)
      {
        // only print lastGPUrad if full time period was covered
        if (lastGPUpos == currentGPUpos)
          {
            std::stringstream last_time_step_str;
            std::stringstream current_time_step_str;
            std::stringstream GPUpos_str;

            last_time_step_str << lastStep;
            current_time_step_str << currentStep;

            for(uint32_t dimIndex=0; dimIndex<simDim; ++dimIndex)
                GPUpos_str << "_" <<currentGPUpos[dimIndex];

            writeFile(radiation->getHostBuffer().getBasePointer(), folderRadPerGPU + "/" + filename_prefix
                      + "_radPerGPU_pos" + GPUpos_str.str()
                      + "_time_" + last_time_step_str.str()
                      + "-" + current_time_step_str.str() + ".dat");
          }
        lastGPUpos = currentGPUpos;
      }

  }


  /** returns number of observers (radiation detectors) */
  static unsigned int elements_amplitude()
  {
    return radiation_frequencies::N_omega * parameters::N_observer; // storage for amplitude results on GPU
  }


  /** combine radiation data from each CPU and store result on master
   *  copyRadiationDeviceToHost() should be called before */
  void collectRadiationOnMaster()
  {
      reduce(nvidia::functors::Add(),
             tmp_result,
             radiation->getHostBuffer().getBasePointer(),
             elements_amplitude(),
             mpi::reduceMethods::Reduce()
             );
  }


  /** add collected radiation data to previously stored data
   *  should be called after collectRadiationOnMaster() */
  void sumAmplitudesOverTime(Amplitude* targetArray, Amplitude* summandArray)
  {
    if (isMaster)
      {
        // add last amplitudes to previous amplitudes
        for (unsigned int i = 0; i < elements_amplitude(); ++i)
          targetArray[i] += summandArray[i];
      }
  }



  /** writes to file the emitted radiation only from the current
   *  time step. Radiation from previous time steps is neglected. */
  void writeLastRadToText()
  {
      // only the master rank writes data
      if (isMaster)
      {
          // write file only if lastRad flag was selected
          if (lastRad)
          {
              // get time step as string
              std::stringstream o_step;
              o_step << currentStep;

              // write lastRad data to txt
              writeFile(tmp_result, folderLastRad + "/" + filename_prefix + "_" + o_step.str() + ".dat");
          }
      }
  }


  /** writes the total radiation (over entire simulation time) to file */
  void writeTotalRadToText()
  {
      // only the master rank writes data
      if (isMaster)
      {
          // write file only if totalRad flag was selected
          if (totalRad)
          {
              // get time step as string
              std::stringstream o_step;
              o_step << currentStep;

              // write totalRad data to txt
              writeFile(timeSumArray, folderTotalRad + "/" + filename_prefix + "_" + o_step.str() + ".dat");
          }
      }
  }


  /** write total radiation data as HDF5 file */
  void writeAmplitudesToHDF5()
  {
      if (isMaster)
      {
          writeHDF5file(timeSumArray, std::string("radiationHDF5/radAmplitudes_"));
      }
  }


  /** perform all operations to get data from GPU to master */
  void collectDataGPUToMaster()
  {
      // collect data GPU -> CPU -> Master
      copyRadiationDeviceToHost();
      collectRadiationOnMaster();
      sumAmplitudesOverTime(timeSumArray, tmp_result);
  }


  /** write all possible/selected output */
  void writeAllFiles(const DataSpace<simDim> currentGPUpos)
  {
      // write data to files
      saveRadPerGPU(currentGPUpos);
      writeLastRadToText();
      writeTotalRadToText();
      writeAmplitudesToHDF5();
  }


  /** This method returns hdf5 data structure names for amplitudes
   *
   *  Arguments:
   *  int index - index of Amplitude
   *              "-1" return record name
   *
   *  Return:
   *  std::string - name
   *
   * This method avoids initializing BOOST_STATIC_CONSTEXPR string arrays.
   */
  static const std::string dataLabels(int index)
  {
      const std::string path("Amplitude/");

      /* return record name if handed -1 */
      if(index == -1)
          return path;

      const std::string dataLabelsList[] = {"x/Re",
                                            "x/Im",
                                            "y/Re",
                                            "y/Im",
                                            "z/Re",
                                            "z/Im"};

      return path + dataLabelsList[index];
  }

  /** This method returns hdf5 data structure names for detector directions
   *
   *  Arguments:
   *  int index - index of detector
   *              "-1" return record name
   *
   *  Return:
   *  std::string - name
   *
   * This method avoids initializing static const string arrays.
   */
  static const std::string dataLabelsDetectorDirection(int index)
  {
      const std::string path("DetectorDirection/");

      /* return record name if handed -1 */
      if(index == -1)
          return path;

      const std::string dataLabelsList[] = {"x",
                                            "y",
                                            "z"};

      return path + dataLabelsList[index];
  }


  /** This method returns hdf5 data structure names for detector frequencies
   *
   *  Arguments:
   *  int index - index of detector
   *              "-1" return record name
   *
   *  Return:
   *  std::string - name
   *
   * This method avoids initializing static const string arrays.
   */
  static const std::string dataLabelsDetectorFrequency(int index)
  {
      const std::string path("DetectorFrequency/");

      /* return record name if handed -1 */
      if(index == -1)
          return path;

      const std::string dataLabelsList[] = {"omega"};

      return path + dataLabelsList[index];
  }



  /** Write Amplitude data to HDF5 file
   *
   * Arguments:
   * Amplitude* values - array of complex amplitude values
   * std::string name - path and beginning of file name to store data to
   */
  void writeHDF5file(Amplitude* values, std::string name)
  {
      splash::SerialDataCollector HDF5dataFile(1);
      splash::DataCollector::FileCreationAttr fAttr;

      splash::DataCollector::initFileCreationAttr(fAttr);
      fAttr.enableCompression = compressionOn;

      std::ostringstream filename;
      filename << name << currentStep;

      HDF5dataFile.open(filename.str().c_str(), fAttr);

      typename PICToSplash<float_64>::type radSplashType;


      splash::Dimensions bufferSize(Amplitude::numComponents,
                                    radiation_frequencies::N_omega,
                                    parameters::N_observer);

      splash::Dimensions componentSize(1,
                                       radiation_frequencies::N_omega,
                                       parameters::N_observer);

      splash::Dimensions stride(Amplitude::numComponents,1,1);

      /* get the radiation amplitude unit */
      Amplitude UnityAmplitude(1., 0., 0., 0., 0., 0.);
      const picongpu::float_64 factor = UnityAmplitude.calc_radiation() * UNIT_ENERGY * UNIT_TIME ;

      for(uint32_t ampIndex=0; ampIndex < Amplitude::numComponents; ++ampIndex)
      {
          splash::Dimensions offset(ampIndex,0,0);
          splash::Selection dataSelection(bufferSize,
                                          componentSize,
                                          offset,
                                          stride);

          /* save data for each x/y/z * Re/Im amplitude */
          HDF5dataFile.write(currentStep,
                             radSplashType,
                             3,
                             dataSelection,
                             (meshesPathName + dataLabels(ampIndex)).c_str(),
                             values);

          /* save SI unit as attribute together with data set */
          HDF5dataFile.writeAttribute(currentStep,
                                      radSplashType,
                                      (meshesPathName + dataLabels(ampIndex)).c_str(),
                                      "unitSI",
                                      &factor);
      }

      /* save SI unit as attribute in the Amplitude group (for convenience) */
      HDF5dataFile.writeAttribute(currentStep,
                                  radSplashType,
                                  (meshesPathName + std::string("Amplitude")).c_str(),
                                  "unitSI",
                                  &factor);

      /* save detector position / observation direction */
      splash::Dimensions bufferSizeDetector(3,
                                            1,
                                            parameters::N_observer);

      splash::Dimensions componentSizeDetector(1,
                                               1,
                                               parameters::N_observer);

      splash::Dimensions strideDetector(3,1,1);

      for(uint32_t detectorDim=0; detectorDim < 3; ++detectorDim)
      {
          splash::Dimensions offset(detectorDim,0,0);
          splash::Selection dataSelection(bufferSizeDetector,
                                      componentSizeDetector,
                                      offset,
                                      strideDetector);

          HDF5dataFile.write(currentStep,
                             radSplashType,
                             3,
                             dataSelection,
                             (meshesPathName + dataLabelsDetectorDirection(detectorDim)).c_str(),
                             detectorPositions);
      }



      /* save detector frequencies */
      splash::Dimensions bufferSizeOmega(1,
                                         radiation_frequencies::N_omega,
                                         1);

      splash::Dimensions strideOmega(1,1,1);

      splash::Dimensions offset(0,0,0);
      splash::Selection dataSelection(bufferSizeOmega,
                                      bufferSizeOmega,
                                      offset,
                                      strideOmega);

      HDF5dataFile.write(currentStep,
                         radSplashType,
                         3,
                         dataSelection,
                         (meshesPathName + dataLabelsDetectorFrequency(0)).c_str(),
                         detectorFrequencies);

      /* save SI unit as attribute together with data set */
      const picongpu::float_64 factorOmega = 1.0 / UNIT_TIME ;
      HDF5dataFile.writeAttribute(currentStep,
                                  radSplashType,
                                  (meshesPathName + dataLabelsDetectorFrequency(0)).c_str(),
                                  "unitSI",
                                  &factorOmega);

      /* begin openPMD attributes */
      /* begin required openPMD attributes */
      std::string openPMDversion("1.0.0");
      splash::ColTypeString ctOpenPMDversion(openPMDversion.length());
      HDF5dataFile.writeGlobalAttribute( ctOpenPMDversion,
                                         "openPMD",
                                         openPMDversion.c_str() );

      const uint32_t openPMDextension = 0; // no extension
      splash::ColTypeUInt32 ctUInt32;
      HDF5dataFile.writeGlobalAttribute( ctUInt32,
                                         "openPMDextension",
                                         &openPMDextension );

      std::string basePath("/data/%T/");
      splash::ColTypeString ctBasePath(basePath.length());
      HDF5dataFile.writeGlobalAttribute(ctBasePath,
                                        "basePath",
                                        basePath.c_str() );

      splash::ColTypeString ctMeshesPath(meshesPathName.length());
      HDF5dataFile.writeGlobalAttribute(ctMeshesPath,
                                        "meshesPath",
                                        meshesPathName.c_str() );


      splash::ColTypeString ctParticlesPath(particlesPathName.length());
      HDF5dataFile.writeGlobalAttribute( ctParticlesPath,
                                         "particlesPath",
                                         particlesPathName.c_str() );

      std::string iterationEncoding("fileBased");
      splash::ColTypeString ctIterationEncoding(iterationEncoding.length());
      HDF5dataFile.writeGlobalAttribute( ctIterationEncoding,
                                         "iterationEncoding",
                                         iterationEncoding.c_str() );

      /* the ..._0_0_0... extension comes from the current filename
         formating of the serial data colector in libSplash */
      const int indexCutDirectory = name.rfind('/');
      std::string iterationFormat(name.substr(indexCutDirectory + 1) +  std::string("%T_0_0_0.h5"));
      splash::ColTypeString ctIterationFormat(iterationFormat.length());
      HDF5dataFile.writeGlobalAttribute( ctIterationFormat,
                                         "iterationFormat",
                                         iterationFormat.c_str() );

      typedef PICToSplash<float_X>::type SplashFloatXType;
      SplashFloatXType splashFloatXType;
      HDF5dataFile.writeAttribute(currentStep, splashFloatXType, NULL, "dt", &DELTA_T);
      const float_X time = float_X(currentStep) * DELTA_T;
      HDF5dataFile.writeAttribute(currentStep, splashFloatXType, NULL, "time", &time);
      splash::ColTypeDouble ctDouble;
      HDF5dataFile.writeAttribute(currentStep, ctDouble, NULL, "timeUnitSI", &UNIT_TIME);

      /* end required openPMD attributes */

      /* begin recommended openPMD attributes */

      std::string author = Environment<>::get().SimulationDescription().getAuthor();
      if( author.length() > 0 )
        {
          splash::ColTypeString ctAuthor(author.length());
          HDF5dataFile.writeGlobalAttribute( ctAuthor,
                                             "author",
                                             author.c_str() );
        }

      std::string software("PIConGPU");
      splash::ColTypeString ctSoftware(software.length());
      HDF5dataFile.writeGlobalAttribute( ctSoftware,
                                         "software",
                                         software.c_str() );

      std::stringstream softwareVersion;
      softwareVersion << PICONGPU_VERSION_MAJOR << "."
                      << PICONGPU_VERSION_MINOR << "."
                      << PICONGPU_VERSION_PATCH;
      splash::ColTypeString ctSoftwareVersion(softwareVersion.str().length());
      HDF5dataFile.writeGlobalAttribute( ctSoftwareVersion,
                                         "softwareVersion",
                                         softwareVersion.str().c_str() );

      std::string date  = getDateString("%F %T %z");
      splash::ColTypeString ctDate(date.length());
      HDF5dataFile.writeGlobalAttribute( ctDate,
                                         "date",
                                         date.c_str() );

      /* end recommended openPMD attributes */
      /* end openPMD attributes */

      HDF5dataFile.close();
    }



  /** Read Amplitude data from HDF5 file
   *
   * Arguments:
   * Amplitude* values - array of complex amplitudes to store data in
   * std::string name - path and beginning of file name with data stored in
   * const int timeStep - time step to read
   */
  void readHDF5file(Amplitude* values, std::string name, const int timeStep)
  {
      splash::SerialDataCollector HDF5dataFile(1);
      splash::DataCollector::FileCreationAttr fAttr;

      splash::DataCollector::initFileCreationAttr(fAttr);

      fAttr.fileAccType = splash::DataCollector::FAT_READ;

      std::ostringstream filename;
      /* add to standard ending added by libSplash for SerialDataCollector */
      filename << name << timeStep << "_0_0_0.h5";

      /* check if restart file exists */
      if( !boost::filesystem::exists(filename.str()) )
      {
          log<picLog::INPUT_OUTPUT > ("Radiation: restart file not found (%1%) - start with zero values") % filename.str();
      }
      else
      {
          HDF5dataFile.open(filename.str().c_str(), fAttr);

          typename PICToSplash<float_64>::type radSplashType;

          splash::Dimensions componentSize(1,
                                           radiation_frequencies::N_omega,
                                           parameters::N_observer);

          const int N_tmpBuffer = radiation_frequencies::N_omega * parameters::N_observer;
          picongpu::float_64* tmpBuffer = new picongpu::float_64[N_tmpBuffer];

          for(uint32_t ampIndex=0; ampIndex < Amplitude::numComponents; ++ampIndex)
          {
              HDF5dataFile.read(timeStep,
                                (meshesPathName + dataLabels(ampIndex)).c_str(),
                                componentSize,
                                tmpBuffer);

              for(int copyIndex = 0; copyIndex < N_tmpBuffer; ++copyIndex)
              {
                  /* convert data directly because Amplitude is just 6 float_64 */
                  ((picongpu::float_64*)values)[ampIndex + Amplitude::numComponents*copyIndex] = tmpBuffer[copyIndex];
              }

          }

          delete[] tmpBuffer;
          HDF5dataFile.close();

          log<picLog::INPUT_OUTPUT > ("Radiation: read radiation data from HDF5");
      }
  }


  /**
   * From the collected data from all hosts the radiated intensity is
   * calculated by calculating the absolute value squared and multiplying
   * this with with the appropriate physics constants.
   * @param values
   * @param name
   */
  void writeFile(Amplitude* values, std::string name)
  {
      std::ofstream outFile;
      outFile.open(name.c_str(), std::ofstream::out | std::ostream::trunc);
      if (!outFile)
      {
          std::cerr << "Can't open file [" << name << "] for output, disable plugin output. " << std::endl;
          isMaster = false; // no Master anymore -> no process is able to write
      }
      else
      {
          for (unsigned int index_direction = 0; index_direction < parameters::N_observer; ++index_direction) // over all directions
          {
              for (unsigned index_omega = 0; index_omega < radiation_frequencies::N_omega; ++index_omega) // over all frequencies
              {
                  // Take Amplitude for one direction and frequency,
                  // calculate the square of the absolute value
                  // and write to file.
                  outFile <<
                    values[index_omega + index_direction * radiation_frequencies::N_omega].calc_radiation() * UNIT_ENERGY * UNIT_TIME << "\t";

              }
              outFile << std::endl;
          }
          outFile.flush();
          outFile << std::endl; //now all data are written to file

          if (outFile.fail())
              std::cerr << "Error on flushing file [" << name << "]. " << std::endl;

          outFile.close();
      }
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
  void calculateRadiationParticles(uint32_t currentStep)
  {
      this->currentStep = currentStep;

      /* the parallelization is ONLY over directions:
       * (a combined parallelization over direction AND frequencies
       * turned out to be slower on GPUs of the Fermi generation (sm_2x) (couple
       * percent) and definitely slower on Kepler GPUs (sm_3x, tested on K20))
       */
      const int N_observer = parameters::N_observer;
      const dim3 gridDim_rad(N_observer);

      /* number of threads per block = number of cells in a super cell
       *          = number of particles in a Frame
       *          (THIS IS PIConGPU SPECIFIC)
       * A Frame is the entity that stores particles.
       * A super cell can have many Frames.
       * Particles in a Frame can be accessed in parallel.
       */

      const dim3 blockDim_rad(PMacc::math::CT::volume<typename MappingDesc::SuperCellSize>::type::value);

      // Some funny things that make it possible for the kernel to calculate
      // the absolute position of the particles
      DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
      const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
      const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
      DataSpace<simDim> globalOffset(subGrid.getLocalDomain().offset);
      globalOffset.y() += (localSize.y() * numSlides);


      // PIC-like kernel call of the radiation kernel
      __cudaKernel(kernelRadiationParticles)
        (gridDim_rad, blockDim_rad)
        (
         /*Pointer to particles memory on the device*/
         particles->getDeviceParticlesBox(),

         /*Pointer to memory of radiated amplitude on the device*/
         radiation->getDeviceBuffer().getDataBox(),
         globalOffset,
         currentStep, *cellDescription,
         freqFkt,
         subGrid.getGlobalDomain().size
         );

      if (dumpPeriod != 0 && currentStep % dumpPeriod == 0)
      {
          collectDataGPUToMaster();
          writeAllFiles(globalOffset);

          // update time steps
          lastStep = currentStep;

          // reset amplitudes on GPU back to zero
          radiation->getDeviceBuffer().reset(false);
      }

  }

};

}



