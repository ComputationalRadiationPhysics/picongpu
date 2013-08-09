/**
 * Copyright 2013 Axel Huebl, Heiko Burau, René Widera, Richard Pausch, Klaus Steiniger
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
 


#ifndef RADIATION_HPP
#define	RADIATION_HPP

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"
#include "basicOperations.hpp"
#include "dimensions/DataSpaceOperations.hpp"

#include "simulation_classTypes.hpp"
#include "mappings/kernel/AreaMapping.hpp"
#include "plugins/IPluginModule.hpp"

#include "plugins/radiation/parameters.hpp"
#include "plugins/radiation/check_consistency.hpp"
#include "plugins/radiation/particle.hpp"
#include "plugins/radiation/amplitude.hpp"
#include "plugins/radiation/calc_amplitude.hpp"

#include "mpi/reduceMethods/Reduce.hpp"
#include "mpi/MPIReduce.hpp"
#include "nvidia/functors/Add.hpp"


#if (__NYQUISTCHECK__==1)
#include "plugins/radiation/nyquist_low_pass.hpp"
#endif

#include "plugins/radiation/radFormFactor.hpp"
#include "simulation_defines/param/observer.hpp"
#include "sys/stat.h"


namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;


///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////  Radiation Kernel  //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

/**
 * The radiation kernel calculates for all particles on the device the 
 * emitted radiation for every direction and every frequency.
 * The parallelization is as follows:
 *  - There are as many Blocks of threads as there are directions for which 
 *    radiation needs to be calculated. (A block of threads shares 
 *    shared memory)
 *  - The number of threads per block is equal to the number of cells per 
 *    super cells which is also equal to the number of particles per frame
 * 
 * The procedure starts with calculating unique ids for the threads and 
 * initializing the shared memory.
 * Then a loop over all super cells starts.
 * Every thread loads a particle from that super cell and calculates its 
 * retarted time and its real amplitude (both is dependent of the direction). 
 * For every Particle 
 * exists therefor a unique space within the shared memory.
 * After that, a thread calculates for a specific frequency the emitted 
 * radiation of all particles. 
 * @param pb
 * @param radiation
 * @param globalOffset
 * @param currentStep
 * @param mapper
 * @param freqFkt
 */
template<class ParBox, class DBox, class Mapping>
__global__
/*__launch_bounds__(256, 4)*/
void kernelRadiationParticles(ParBox pb,
                              DBox radiation,
                              DataSpace<simDim> globalOffset,
                              uint32_t currentStep,
                              Mapping mapper,
                              radiation_frequencies::FreqFunctor freqFkt)
{

    typedef typename MappingDesc::SuperCellSize Block;
    typedef typename ParBox::FrameType FRAME;

    __shared__ FRAME *frame; // pointer to  frame storing particles
    __shared__ bool isValid; // bool saying if frame is valid
    __shared__ lcellId_t particlesInFrame; // number  of paricles in current frame 

    using namespace parameters; // parameters of radiation

    /// calculate radiated Amplitude
    /* parallelized in 1 dimensions: 
     * looking direction (theta) 
     * (not anymore data handling)
     * create shared memory for particle data to reduce global memory calls
     * every thread in a block loads one particle and every thread runs
     * through all particles and calculates the radiation for one direction
     * for all frequencies
     */

    // vectorial part of the integrand in the Jackson formula
    __shared__ vec2 real_amplitude_s[Block::elements];

    // retarded time
    __shared__ numtype2 t_ret_s[Block::elements];

    // storage for macro particle weighting needed if 
    // the coherent and incoherent radiation of a single
    // macro-particle needs to be considered
#if (__COHERENTINCOHERENTWEIGHTING__==1)
    __shared__ float_X radWeighting_s[Block::elements];
#endif

    // particle counter used if not all particles are considered for 
    // radiation calculation
    __shared__ int counter_s;

    // memory for Nyquist frequency at current time step
#if (__NYQUISTCHECK__==1) 
    __shared__ NyquistLowPass lowpass_s[Block::elements];
#endif


    const int theta_idx = blockIdx.x; //blockIdx.x is used to determine theata
    const uint32_t linearThreadIdx = threadIdx.x; // used for determine omega and particle id
    // old:  DataSpaceOperations<DIM3>::map<Block > (DataSpace<DIM3 > (threadIdx));


    __syncthreads(); /*wait that all shared memory is initialised*/


    // simulation time (needed for retarded time)
    const numtype2 t((numtype2) currentStep * (numtype2) DELTA_T);

    // looking direction (needed for observer) used in the thread
    const vec2 look = radiation_observer::observation_direction(theta_idx);

    // get extent of guarding super cells (needed to ignore them)
    const int guardingSuperCells = mapper.getGuardingSuperCells();


    // number of super cells on GPU
    const DataSpace<DIM3> superCellsCount(mapper.getGridSuperCells());

    // go over all super cells on GPU 
    // but ignore all guarding supercells
    for (int z = guardingSuperCells; z < superCellsCount.z() - guardingSuperCells; ++z)
        for (int y = guardingSuperCells; y < superCellsCount.y() - guardingSuperCells; ++y)
            for (int x = guardingSuperCells; x < superCellsCount.x() - guardingSuperCells; ++x)
            {
                /* warpId != 1 synchronization is needed, 
                   since a racecondition can occure if "continue loop" is called, 
                   all threads must wait for the selection of a new frame 
                   untill all threads have evaluated "isValid"
                 */
                __syncthreads();

                const DataSpace<DIM3> superCell(x, y, z); // select SuperCell
                const DataSpace<simDim> superCellOffset(globalOffset
                                                        + ((superCell - guardingSuperCells)
                                                           * Block::getDataSpace()));
                // -guardingSuperCells remove guarding block

                /*
                 * The Master process (thread 0) in every thread block is in 
                 * charge of loading a Frame from 
                 * the current super cell and evaluate the total number of 
                 * particles in this frame.
                 */
                if (linearThreadIdx == 0)
                {
                    // set frame pointer
                    frame = &(pb.getLastFrame(superCell, isValid));

                    // number of particles in this frame
                    particlesInFrame = pb.getSuperCell(superCell).getSizeLastFrame();

                    counter_s = 0;
                }

                __syncthreads();

                /* goto next supercell 
                 * 
                 * if "isValid" is false then there is no frames 
                 * inside the superCell (anymore)
                 */
                while (isValid)
                {
                    // only threads with particles are running 
                    if (linearThreadIdx < particlesInFrame)
                    {

                        /* initializes "saveParticleAt" flag with -1
                         * because "counter_s" wil never be -1
                         * therfore, if a particle is saved, a value of counter
                         * is stored in "saveParticleAt" != -1 
                         * THIS IS ACTUALLY ONLY NEEDED IF: the radiation flag was set
                         * LATER: can this be optimized? 
                         */
                        int saveParticleAt = -1;

                        /* if radiation is not calculated for all particles
                         * but not via the gamma filter, check which particles
                         * have to be used for radiation calculation
                         */
#if(RAD_MARK_PARTICLE>1) || (RAD_ACTIVATE_GAMMA_FILTER!=0)
                        if (frame->getRadiationFlag()[linearThreadIdx])
#endif
                            saveParticleAt = atomicAdd(&counter_s, 1);
                        /* for information:
                         *   atomicAdd returns an int with the previous 
                         *   value of "counter_s" != -1
                         *   therefore, if a particle is selected
                         *   "saveParticleAs" != -1 
                         */

                        // if a particle needs to be considered
                        if (saveParticleAt != -1)
                        {

                            // calculate global position 
                            lcellId_t cellIdx = frame->getCellIdx()[linearThreadIdx];

                            // position inside of the cell 
                            float3_X pos = frame->getPosition()[linearThreadIdx];

                            // calculate global position of cell
                            const DataSpace<DIM3> globalPos(superCellOffset
                                                            + DataSpaceOperations<DIM3>::template map<Block >
                                                            (cellIdx));

                            // add global position of cell with local position of particle in cell
                            const vec1 particle_locationNow = vec1(
                                                                   ((float_X) globalPos.x() + (float_X) pos.x()) * CELL_WIDTH,
                                                                   ((float_X) globalPos.y() + (float_X) pos.y()) * CELL_HEIGHT,
                                                                   ((float_X) globalPos.z() + (float_X) pos.z()) * CELL_DEPTH);

                            // get old and new particle momenta
                            const vec1 particle_momentumNow = vec1(frame->getMomentum()[linearThreadIdx]);
                            const vec1 particle_momentumOld = vec1(frame->getMomentum_mt1()[linearThreadIdx]);


                            /* get macro-particle weighting
                             * 
                             * Info:
                             * the weighting is the number of real particles described 
                             * by a macro-particle 
                             */
                            const float_X weighting = frame->getWeighting()[linearThreadIdx];

                            /* only of coherent and incoherent radiation of a sibgle macro-particle is
                             * considered, the weighting of each macro-particle needs to be stored
                             * in order to be considered when the actual frequency calulation is done
                             */
#if (__COHERENTINCOHERENTWEIGHTING__==1)
                            radWeighting_s[saveParticleAt] = weighting;
#endif

                            // mass of macro-particle
                            const float_X particle_mass = frame->getMass(weighting);


                            /****************************************************
                             **** Here happens the true physical calculation ****
                             ****************************************************/

                            // set up particle using the radiation onw's particle class
                            const Particle particle(particle_locationNow,
                                                    particle_momentumOld,
                                                    particle_momentumNow,
                                                    particle_mass);

                            // set up amplitude calculator 
                            typedef Calc_Amplitude< Retarded_time_1, Old_DFT > Calc_Amplitude_n_sim_1;

                            // calculate amplitude
                            const Calc_Amplitude_n_sim_1 amplitude3(particle,
                                                                    DELTA_T,
                                                                    t);


                            // if coherent and incoherent of single macro-particle is considered
#if (__COHERENTINCOHERENTWEIGHTING__==1)
                            // get charge of single electron ! (weighting=1.0f)
                            const picongpu::float_X particle_charge = frame->getCharge(1.0f);

                            // compute real amplitude of macro-particle with a charge of 
                            // a single electron 
                            real_amplitude_s[saveParticleAt] = amplitude3.get_vector(look) *
                                particle_charge *
                                (numtype2) DELTA_T;
#else
                            // if coherent and incoherent of single macro-particle is NOT considered

                            // get charge of entire macro-particle
                            const picongpu::float_X particle_charge = frame->getCharge(weighting);

                            // compute real amplitude of macro-particle
                            real_amplitude_s[saveParticleAt] = amplitude3.get_vector(look) *
                                particle_charge *
                                (numtype2) DELTA_T;
#endif

                            // retarded time stored in shared memory 
                            t_ret_s[saveParticleAt] = amplitude3.get_t_ret(look);

                            // if Nyquist-limter is used, then the NyquistLowPlass object
                            // is setup and stored in shared memory
#if (__NYQUISTCHECK__==1)
                            lowpass_s[saveParticleAt] = NyquistLowPass(look, particle);
#endif


                        } // END: if a particle needs to be considered
                    } // END: only threads with particles are running 


                    __syncthreads(); // wait till every thread has loaded its particle data



                    // run over all ony valid omegas for this thread
                    for (int o = linearThreadIdx; o < radiation_frequencies::N_omega; o += Block::elements)
                    {

                        /* storage for amplitude (complex 3D vector)
                         * it  is inizialized with zeros (  0 +  i 0 )
                         */
                        Amplitude amplitude = Amplitude::zero();

                        // compute frequency "omega" using for-loop-index "o" 
                        const numtype2 omega = freqFkt(o);


                        // if coherent and incoherent radiation of a single macro-particle 
                        // is considered, creare a form factor object
#if (__COHERENTINCOHERENTWEIGHTING__==1)
                        const radFormFactor_selected::radFormFactor myRadFormFactor;
#endif

                        /* Particle loop: thread runs through loaded particle data
                         *
                         * Summation of Jackson radiation formula integrand 
                         * over all electrons for fixed, thread-specific 
                         * frequency 
                         */
                        for (int j = 0; j < counter_s; ++j)
                        {

                            // if Nyquist-limiter is on
#if (__NYQUISTCHECK__==1)
                            // check Nyquist-limit for each particle "j" and each frequeny "omega"
                            if (lowpass_s[j].check(omega))
                            {
#endif

                                /****************************************************
                                 **** Here happens the true physical calculation ****
                                 ****************************************************/


                                // if coherent/incoherent radiation of single macro-particle 
                                // is considered
                                // the form factor influences the real amplitude
#if (__COHERENTINCOHERENTWEIGHTING__==1)
                                const vec2 weighted_real_amp = real_amplitude_s[j] * typeCast<float_64 >
                                    (myRadFormFactor(radWeighting_s[j], omega, look));
#else
                                // if coherent/incoherent radiation of single macro-particle 
                                // is NOT considered
                                // no change on real amplitude is performed
                                const vec2 weighted_real_amp = real_amplitude_s[j];
#endif

                                // complex amplitude for j-th particle
                                Amplitude amplitude_add(weighted_real_amp,
                                                        t_ret_s[j] * omega);

                                // add this single amplitude those previously considered
                                amplitude += amplitude_add;

                                // if Nyquist limietr is on
#if (__NYQUISTCHECK__==1)
                            }// END: check Nyquist-limit for each particle "j" and each frequeny "omega"
#endif

                        }// END: Particle loop


                        /* the radiation contribution of the following is added to global memory:
                         *     - valid particles of last super cell
                         *     - from this (one) time step
                         *     - omega_id = theta_idx * radiation_frequencies::N_omega + o
                         */
                        radiation[theta_idx * radiation_frequencies::N_omega + o] += amplitude;


                    } // end frequency loop


                    // wait till all radiation contributions for this super cell are done
                    __syncthreads();



                    if (linearThreadIdx == 0)
                    {
                        /* First threads starts loading next frame of the super-cell:
                         *
                         * Info:
                         *   The Calculation starts with the last super cell, all 
                         *   other super cells before that are full and 
                         *   therefore have Block::elements (=256) number of 
                         *   particles
                         */
                        particlesInFrame = Block::elements;
                        frame = &(pb.getPreviousFrame(*frame, isValid));
                        counter_s = 0;
                    }

                    // wait till first thread has loaded new frame
                    __syncthreads();

                    // run thrue while-loop(is Valid) again

                } // end while(isValid)

            } // end loop over all super cells


} // end radiation kernel





///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////  Radiation Analyzer Class  ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template<class ParticlesType>
class Radiation : public ISimulationIO, public IPluginModule
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

    std::string analyzerName;
    std::string analyzerPrefix;
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

    bool isMaster;

    uint32_t currentStep;
    uint32_t lastStep;

    bool radRestart;
    std::string pathRestart;

    mpi::MPIReduce reduce;

public:

    Radiation(std::string name, std::string prefix) :
    analyzerName(name),
    analyzerPrefix(prefix),
    filename_prefix(name),
    particles(NULL),
    radiation(NULL),
    cellDescription(NULL),
    notifyFrequency(0),
    dumpPeriod(0),
    totalRad(false),
    lastRad(false),
    timeSumArray(NULL),
    isMaster(false),
    currentStep(0),
    radPerGPU(false),
    lastStep(0),
    radRestart(false)
    {

        ModuleConnector::getInstance().registerModule(this);
    }

    virtual ~Radiation()
    {

    }

    /**
     * This function represents what is actually calculated if the analyzer 
     * is called. Here, one only sets the particles pointer to the data of 
     * the latest time step and calls the 'calculateRadiationParticles' 
     * function if for the actual time step radiation is to be calculated. 
     * @param currentStep
     */
    void notify(uint32_t currentStep)
    {

        DataConnector &dc = DataConnector::getInstance();

        particles = &(dc.getData<ParticlesType > ((uint32_t) ParticlesType::FrameType::CommunicationTag, true));

        if (currentStep >= radStart)
        {
            // radEnd = 0 is default, calculates radiation until simulation
            // end
            if (currentStep <= radEnd || radEnd == 0)
            {
                log<radLog::SIMULATION_STATE > ("radiation gets calculated: timestep %1% ") % currentStep;

                /* CORE + BORDER is PIC black magic, currently not needed
                 * 
                 */
                calculateRadiationParticles < CORE + BORDER > (currentStep);

                log<radLog::SIMULATION_STATE > ("radiation got calculated: timestep %1% ") % currentStep;
            }
        }
    }

    void moduleRegisterHelp(po::options_description& desc)
    {

        desc.add_options()
            ((analyzerPrefix + ".period").c_str(), po::value<uint32_t > (&notifyFrequency), "enable analyser [for each n-th step]")
            ((analyzerPrefix + ".dump").c_str(), po::value<uint32_t > (&dumpPeriod)->default_value(0), "dump integrated radiation from last dumped step [for each n-th step] (0 = only print data at end of simulation)")
            ((analyzerPrefix + ".lastRadiation").c_str(), po::value<bool > (&lastRad)->default_value(false), "enable(1)/disable(0) calculation integrated radiation from last dumped step")
            ((analyzerPrefix + ".folderLastRad").c_str(), po::value<std::string > (&folderLastRad)->default_value("lastRad"), "folder in which the integrated radiation from last dumped step is written")
            ((analyzerPrefix + ".totalRadiation").c_str(), po::value<bool > (&totalRad)->default_value(false), "enable(1)/disable(0) calculation integrated radiation from start of simulation")
            ((analyzerPrefix + ".folderTotalRad").c_str(), po::value<std::string > (&folderTotalRad)->default_value("totalRad"), "folder in which the integrated radiation from start of simulation is written")
            ((analyzerPrefix + ".start").c_str(), po::value<uint32_t > (&radStart)->default_value(2), "time index when radiation should start with calculation")
            ((analyzerPrefix + ".end").c_str(), po::value<uint32_t > (&radEnd)->default_value(0), "time index when radiation should end with calculation")
            ((analyzerPrefix + ".omegaList").c_str(), po::value<std::string > (&pathOmegaList)->default_value("_noPath_"), "path to file containing all frequencies to calculate")
            ((analyzerPrefix + ".radPerGPU").c_str(), po::value<bool > (&radPerGPU)->default_value(false), "enable(1)/disable(0) radiation output from each GPU individually")
            ((analyzerPrefix + ".folderRadPerGPU").c_str(), po::value<std::string > (&folderRadPerGPU)->default_value("radPerGPU"), "folder in which the radiation of each GPU is written")
            ((analyzerPrefix + ".restart").c_str(), po::value<bool > (&radRestart)->default_value(false), "enable(1)/disable(0) restart flag");
    }

    std::string moduleGetName() const
    {

        return analyzerName;
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {

        this->cellDescription = cellDescription;
    }

private:

    /**
     * The module is loaded on every host pc, and therefor this function is 
     * executed on every host pc.
     * One host with MPI rank 0 is defined to be the master.
     * It creates a folder where all the 
     * results are saved and, depending on the type of radiation calculation,
     * creates an additional data structure for the summation of all 
     * intermediate values.
     * On every host data structure for storage of the calculated radiation 
     * is created.       */
    void moduleLoad()
    {
        if (notifyFrequency > 0)
        {
            /*only rank 0 create a file*/
            isMaster = reduce.hasResult(mpi::reduceMethods::Reduce());
            const int elements_amplitude = radiation_frequencies::N_omega * parameters::N_theta; // storage for amplitude results on GPU

            radiation = new GridBuffer<Amplitude, DIM1 > (DataSpace<DIM1 > (elements_amplitude)); //create one int on gpu und host

            freqInit.Init(pathOmegaList);
            freqFkt = freqInit.getFunctor();


            DataConnector::getInstance().registerObserver(this, notifyFrequency);

            if (isMaster && totalRad)
            {
                mkdir("radRestart", 0755);
            }


            if (isMaster && radPerGPU)
            {
                mkdir((folderRadPerGPU).c_str(), 0755);
            }

            if (isMaster && totalRad)
            {
                //create folder for total output
                mkdir((folderTotalRad).c_str(), 0755);
                timeSumArray = new Amplitude[elements_amplitude];
                for (int i = 0; i < elements_amplitude; ++i)
                    timeSumArray[i] = Amplitude();
            }
            if (isMaster && lastRad)
            {
                //create folder for total output

                mkdir((folderLastRad).c_str(), 0755);
            }

        }
    }

    void moduleUnload()
    {
        if (notifyFrequency > 0)
        {

            // Some funny things that make it possible for the kernel to calculate
            // the absolut position of the particles
            PMACC_AUTO(simBox, SubGrid<simDim>::getInstance().getSimulationBox());
            DataSpace<simDim> localSize(simBox.getLocalSize());
            VirtualWindow window(MovingWindow::getInstance().getVirtualWindow(currentStep));
            DataSpace<simDim> globalOffset(simBox.getGlobalOffset());
            globalOffset.y() += (localSize.y() * window.slides);

            //only print data at end of simulation
            if (dumpPeriod == 0)
                combineData(globalOffset);
            if (isMaster && totalRad)
            {
                delete[] timeSumArray;
            }

            if (radiation)
                delete radiation;
            CUDA_CHECK(cudaGetLastError());
        }
    }

    /**
     * This function is called by the calculateRadiationParticles() function
     * if storing of intermediate results is activated (dumpPeriod != 0) 
     * otherwise it is invoked by moduleUnload().
     * 
     * On every host the calculated radiation (radiation from the particles
     * on that device for all directions and frequencies) is transferred 
     * from the gpu to the host, and then the data from all hosts is 
     * combined on the master host. Hence, the emitted radiation of all 
     * particles for every direction and step is available on the master 
     * host. 
     */
    void combineData(const DataSpace<simDim> currentGPUpos)
    {

        const unsigned int elements_amplitude = radiation_frequencies::N_omega * parameters::N_theta; // storage for amplitude results on GPU
        Amplitude *result = new Amplitude[elements_amplitude];


        radiation->deviceToHost();
        __getTransactionEvent().waitForFinished();

        if (radPerGPU)
        {
            if (lastGPUpos == currentGPUpos)
            {
                std::stringstream last_time_step_str;
                std::stringstream current_time_step_str;
                std::stringstream GPUposX;
                std::stringstream GPUposY;
                std::stringstream GPUposZ;


                last_time_step_str << lastStep;
                current_time_step_str << currentStep;
                GPUposX << currentGPUpos.x();
                GPUposY << currentGPUpos.y();
                GPUposZ << currentGPUpos.z();

                writeFile(radiation->getHostBuffer().getBasePointer(), folderRadPerGPU + "/" + filename_prefix
                          + "_radPerGPU_pos_" + GPUposX.str() + "_" + GPUposY.str() + "_" + GPUposZ.str()
                          + "_time_" + last_time_step_str.str() + "-" + current_time_step_str.str() + ".dat");
            }
        }

        reduce(nvidia::functors::Add(),
               result,
               radiation->getHostBuffer().getBasePointer(),
               elements_amplitude,
               mpi::reduceMethods::Reduce()
               );

        if (isMaster)
        {
            std::stringstream o_step;
            o_step << currentStep;
            if (totalRad)
            {
                /*
                 * totalRad writes to file the total emitted radiation up
                 * to the current time step.
                 */

                if (radRestart)
                {
                    std::stringstream o_bu_step;
                    o_bu_step << currentStep - dumpPeriod;

                    loadBackup(timeSumArray, std::string("radRestart") + "/" + std::string("radRestart") + "_" + o_bu_step.str() + ".dat");
                    radRestart = false; // reset restart flag
                }


                for (int i = 0; i < elements_amplitude; ++i)
                    timeSumArray[i] += result[i];
                writeFile(timeSumArray, folderTotalRad + "/" + filename_prefix + "_" + o_step.str() + ".dat");
                writeBackup(timeSumArray, std::string("radRestart") + "/" + std::string("radRestart") + "_" + o_step.str() + ".dat");
            }

            if (lastRad)
                /*
                 * lastRad writes to file the emitted radiation only from
                 * the current time step. That is, radiation from previous
                 * time steps is neglected.
                 */
                writeFile(result, folderLastRad + "/" + filename_prefix + "_" + o_step.str() + ".dat");
        }

        delete[] result;

        lastStep = currentStep;
        lastGPUpos = currentGPUpos;
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
            std::cerr << "Can't open file [" << name << "] for output, diasble analyser output. " << std::endl;
            isMaster = false; // no Master anymore -> no process is able to write
        }
        else
        {
            for (unsigned int index_direction = 0; index_direction < parameters::N_theta; ++index_direction) // over all directions
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

    void writeBackup(Amplitude* values, std::string name)
    {

        std::ofstream outFile;
        outFile.open(name.c_str(), std::ofstream::out | std::ofstream::binary);
        if (!outFile)
        {
            std::cerr << "Can't open file [" << name << "] for backup, diasble analyser output. " << std::endl;
            isMaster = false; // no Master anymore -> no process is able to write
        }
        else
        {
            outFile.write((char*) values, sizeof (Amplitude) * parameters::N_theta * radiation_frequencies::N_omega);
        }

        outFile.close();
    }

    void loadBackup(Amplitude* values, std::string name)
    {

        std::ifstream inFile;
        inFile.open(name.c_str(), std::ifstream::in | std::ifstream::binary);
        if (!inFile)
        {
            std::cerr << "Can't open file [" << name << "] for loading backup. " << std::endl;
        }
        else
        {
            inFile.read((char*) values, sizeof (Amplitude) * parameters::N_theta * radiation_frequencies::N_omega);
            std::cout << "Radiation: backup files have been loaded." << std::endl;
        }

        inFile.close();
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
         * (a combinded parallelization over direction AND frequencies 
         *  turned out to be slower on fermis (couple percent) and 
         *  definitly slower on kepler k20)
         */
        const int N_theta = parameters::N_theta;
        const dim3 gridDim_rad(N_theta);

        /* number of threads per block = number of cells in a super cell
         *          = number of particles in a Frame 
         *          (THIS IS PIConGPU SPECIFIC)
         * A Frame is the entity that stores particles. 
         * A super cell can have many Frames. 
         * Particles in a Frame can be accessed in parallel.
         * It has a fixed size of 256.
         */

        const dim3 blockDim_rad(MappingDesc::SuperCellSize::elements);


        // std::cout<<"Grid: "<<gridDim_rad.x()<<" "<<gridDim_rad.y()<<" "<<gridDim_rad.z()<<std::endl;
        // std::cout<<"Block: "<<blockDim_rad.x()<<" "<<blockDim_rad.y()<<" "<<blockDim_rad.z()<<std::endl;


        // Some funny things that make it possible for the kernel to calculate
        // the absolut position of the particles
        DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
        VirtualWindow window(MovingWindow::getInstance().getVirtualWindow(currentStep));
        DataSpace<simDim> globalOffset(SubGrid<simDim>::getInstance().getSimulationBox().getGlobalOffset());
        globalOffset.y() += (localSize.y() * window.slides);

        // PIC-like kernel call of the radiation kernel
        __cudaKernel(kernelRadiationParticles)
            (gridDim_rad, blockDim_rad)
            (
             /*Pointer to particles memory on the device*/
             particles->getDeviceParticlesBox(),

             /*Pointer to memory of radiated amplitude on the device*/
             radiation->getDeviceBuffer().getDataBox(),
             globalOffset,
             currentStep, *cellDescription, freqFkt
             );

        if (dumpPeriod != 0 && currentStep % dumpPeriod == 0)
        {
            combineData(globalOffset);
            radiation->getDeviceBuffer().reset(false);
        }

    }

};

}

#endif	/* RADIATION_HPP */

