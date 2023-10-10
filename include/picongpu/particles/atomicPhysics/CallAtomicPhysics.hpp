/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov,
 *                     Brian Marre
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

#include "picongpu/param/atomicPhysics.param"
#include "picongpu/particles/atomicPhysics/AtomicData.hpp"
#include "picongpu/particles/atomicPhysics/AtomicPhysics.kernel"

#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>


/**@file
 * This file implements the CallAtomicPhyiscs functor called by the AtomicPhysics stage
 *      see <include/picongpu/simulation/stage/AtomicPhysics.hpp>
 * This functor reads and stores the atomic input data for atomic physics on CPU for
 *  the given species and calls the device kernel.
 *
 * One instance of this class is stored as a private member of by the AtomicPhysics
 *  stage
 *
 * @tparam T_IonSpecies ... species of macro particle to apply atomic physics to.
 *
 * @private-member:
 * uint numberStates        ... number of states included in atomic data
 * uint numberTransitions   ... number of transitions included in input data
 * "pointer" atomicData     ... pointer to atomic data
 *
 * @public-member:
 * "uint" protonNumber  ... proton number of Species
 *      in input data set
 * "type" IonSpecies    ... ion species we apply atomic physics to
 *
 * methods:
 * "List of Tupel" readStateData( filename )
 *      reader method for atomic data files, returns List of tupels
 *      (state ID, state energy in eV) taken from file <filename>
 *
 *      input data files can be generated from FlyCHK atomic data files "atomic.inp.<Z>"
 *          Z being the atomic number of the element
 *      using my script "ExtractAtomicData.py", requires flylite installation,
 *          ask me for installation, Brian Marre
 *      @TODO add ability to limit to n_max, 25.12.2020-Brian Marre
 *
 * "List of Tupel" readTransitionData( filename )
 *      reader method for atomic data files, same as readStateData but returns tupels
 *      (lower state ID, higher state ID, oscillator stength, gaunt coeffcients 1-5)
 *
 * void CallAtomicPhysics()
 *      constructor, reads atomic data input files <stateData> and <transitionData>
 *      stores them in cpu and device memory, intended to be called only once per
 *      simulation
 *
 * void operator()
 *      time step functor, is called once per time step by atomic physics stage
 *      calls the atomicPhysicsKernel for every super cell
 *
 */

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            // Functor to apply all operations for atomic physics for a given ion species
            template<typename T_IonSpecies>
            struct CallAtomicPhysics
            {
            public:
                //{ Datatypes for later acess
                //{ ions
                // extract ion species type
                using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;
                // ion species frame type
                using IonFrameType = typename IonSpecies::FrameType;
                // @TODO: get this from species
                //} ions

                //{ atomic state
                // data type used for configNumber
                // TODO: read from type of ion
                using T_ConfigNumberDataType = picongpu::atomicPhysics::configNumberDataType;

                static constexpr uint8_t protonNumber = picongpu::atomicPhysics::protonNumber;
                //} atomic state

                //{ electrons
                // Define electron species type
                using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<
                    VectorAllSpecies,
                    typename pmacc::particles::traits::ResolveAliasFromSpecies<
                        IonSpecies,
                        atomicPhysicsSolver<> /// atomic physics flag of species from species.param file
                        >::type>;
                // electron frame type
                using ElectronFrameType = typename ElectronSpecies::FrameType;
                //} electrons

                //{ input data
                // define std:vector of entries of atomic data sets
                using States = typename std::vector<std::pair<
                    T_ConfigNumberDataType, // index of atomic state
                    float_X // state energy relative to ground state of its ionization state
                            // in eV
                    >>;
                // same as States but for Transitions
                using Transitions = typename std::vector<std::tuple<
                    T_ConfigNumberDataType, // lower state config number
                    T_ConfigNumberDataType, // higher state config number
                    float_X, // collisional oscillator strength
                    float_X, // cinx1, gaunt coefficent used by formula
                    float_X, // cinx2, gaunt coefficent used by formula
                    float_X, // cinx3, gaunt coefficent used by formula
                    float_X, // cinx4, gaunt coefficent used by formula
                    float_X, // cinx5, gaunt coefficent used by formula
                    float_X // absorption oscillator strength
                    >>;
                //} input data
                //} Datatypes for later access

            private:
                // number of states in input data file
                uint32_t numberStates;
                // number of transitions in input data file
                uint32_t numberTransitions;
                // memory space for pointer to atomic data
                std::unique_ptr<AtomicData<protonNumber, T_ConfigNumberDataType>> atomicData;

            public:
                static States readStateData(std::string fileName)
                {
                    /** read Data method for atomic states
                     *
                     * BEWARE: does not convert to internal units
                     */
                    std::ifstream file(fileName);
                    if(!file)
                    {
                        // data could not be found
                        throw std::runtime_error(
                            "Atomic physics error: could not open atomic data file" + fileName
                            + ", as specifed in CallAtomicPhysics.hpp");
                        return States{};
                    }

                    States result;
                    double stateConfigNumber; // @TODO: catch overflow if full uint64 is used
                    float_X energyOverGroundState;

                    while(file >> stateConfigNumber >> energyOverGroundState)
                    {
                        T_ConfigNumberDataType idx = static_cast<T_ConfigNumberDataType>(stateConfigNumber);
                        auto item = std::make_pair(idx, energyOverGroundState);
                        result.push_back(item);
                    }
                    return result;
                }

                static Transitions readTransitionData(std::string fileName)
                {
                    /** read Data method for atomic transitions
                     *
                     * BEWARE: does not convert to internal units
                     */
                    std::ifstream file(fileName);
                    if(!file)
                    {
                        throw std::runtime_error(
                            "Atomic physics error: could not open atomic data file" + fileName
                            + ", as specifed in CallAtomicPhysics.hpp");
                        return Transitions{};
                    }

                    double idxLower;
                    double idxUpper;
                    float_X collisionalOscillatorStrength;

                    // gauntCoeficients
                    float_X cinx1;
                    float_X cinx2;
                    float_X cinx3;
                    float_X cinx4;
                    float_X cinx5;

                    float_X absorptionOscillatorStrength;

                    Transitions result;

                    while(file >> idxLower >> idxUpper >> collisionalOscillatorStrength >> cinx1 >> cinx2 >> cinx3
                          >> cinx4 >> cinx5 >> absorptionOscillatorStrength)
                    {
                        uint64_t stateIndexLower = static_cast<uint64_t>(idxLower);
                        uint64_t stateIndexUpper = static_cast<uint64_t>(idxUpper);

                        // protection against circle transitions
                        if(stateIndexLower == stateIndexUpper)
                        {
                            printf("ERROR: circular transitions are not supported, treat steps seperately\n");
                            continue;
                        }

                        auto item = std::make_tuple(
                            stateIndexLower,
                            stateIndexUpper,
                            collisionalOscillatorStrength,
                            cinx1,
                            cinx2,
                            cinx3,
                            cinx4,
                            cinx5,
                            absorptionOscillatorStrength);
                        // append to vector result
                        result.push_back(item);
                    }
                    return result;
                }

                CallAtomicPhysics()
                {
                    /** Constructor, loads atomic input data, stores, converts
                     *
                     * BEWARE: prototype implementation ONLY,
                     *  - filenames are hardcoded
                     *  - only a single file can be loaded for all species
                     *  - assumes block form for transitions
                     * @TODO get out of param files seperate for each species, Brian Marre 2020
                     */

                    // read in atomic data:
                    // levels
                    auto levelDataItems
                        = readStateData(picongpu::atomicPhysics::stateDataFileName); // TODO: make file species
                                                                                     // dependent, Brian Marre, 2021
                    // transitions
                    auto transitionDataItems = readTransitionData(
                        picongpu::atomicPhysics::transitionDataFileName); // TODO: make file species dependent, Brian
                                                                          // Marre, 2021

                    // check whether read was sucessfull
                    if(levelDataItems.empty())
                    {
                        std::cout << "Could not read the atomic level data. Check given filename.\n";
                        return;
                    }
                    if(transitionDataItems.empty())
                    {
                        std::cout << "Could not read the atomic transition data. Check given filename.\n";
                        return;
                    }

                    // get number of states in input data set
                    this->numberStates = levelDataItems.size();
                    // number of transitions
                    this->numberTransitions = transitionDataItems.size();

                    // create atomic Data storage class on host and store pointer as private member
                    atomicData = std::make_unique<AtomicData<protonNumber, T_ConfigNumberDataType>>(
                        levelDataItems.size(),
                        transitionDataItems.size());

                    // get acess to data box on host side
                    // on init is empty
                    auto atomicDataHostBox = atomicData->getHostDataBox(
                        0u, // numberStates
                        0u // numberTransitions
                    );

                    // fill atomic level data into dataBox
                    for(uint32_t i = 0; i < levelDataItems.size(); i++)
                    {
                        atomicDataHostBox.addLevel(levelDataItems[i].first, levelDataItems[i].second);
                    }

                    // fill atomic transition data into dataBox
                    for(uint32_t i = 0; i < transitionDataItems.size(); i++)
                    {
                        atomicDataHostBox.addTransition(
                            std::get<0>(transitionDataItems[i]),
                            std::get<1>(transitionDataItems[i]),
                            std::get<2>(transitionDataItems[i]),
                            std::get<3>(transitionDataItems[i]),
                            std::get<4>(transitionDataItems[i]),
                            std::get<5>(transitionDataItems[i]),
                            std::get<6>(transitionDataItems[i]),
                            std::get<7>(transitionDataItems[i]),
                            std::get<8>(transitionDataItems[i]));
                    }

                    // copy data to device buffer
                    atomicData->syncToDevice();
                }

                void operator()(
                    uint32_t const step, // step index
                    MappingDesc const cellDescription) const
                {
                    /** time step functor, calls device kernel AtomicPhysicsKernel defined in
                     *   <picongpu/particles/atomicPhysics/AtomicPhysics.kernel>
                     *
                     * called by Atomic Physics stage operator() once per time step
                     */
                    using namespace pmacc;

                    // ?                  ? should be documented
                    DataConnector& dc = Environment<>::get().DataConnector();

                    // ions
                    auto& ions = *dc.get<IonSpecies>(IonFrameType::getName());

                    // electrons
                    auto& electrons = *dc.get<ElectronSpecies>(ElectronFrameType::getName());

                    // area to apply kernel to
                    AreaMapping<
                        CORE + BORDER, // full local domain, no guards
                        MappingDesc>
                        mapper(cellDescription);

                    // renaming of Kernel, basic construct defined in
                    //   <picongpu/particles/atomicPhysics/AtomicPhysics.kernel>
                    using Kernel = AtomicPhysicsKernel<picongpu::atomicPhysics::maxNumBins>;
                    auto kernel = Kernel{RngFactoryInt{step}, RngFactoryFloat{step}};

                    auto workerCfg = lockstep::makeWorkerCfg<IonFrameType::frameSize>();

                    // macro for call of kernel, once for every super cell
                    PMACC_LOCKSTEP_KERNEL(kernel, workerCfg)
                    (mapper.getGridDim() // how many blocks = how many supercells in local domain

                     )(electrons.getDeviceParticlesBox(),
                       ions.getDeviceParticlesBox(),
                       mapper,
                       atomicData->getDeviceDataBox(this->numberStates, this->numberTransitions),
                       picongpu::atomicPhysics::initialGridWidth, // unit: ATOMIC_UNIT_ENERGY
                       picongpu::atomicPhysics::relativeErrorTarget, // unit: 1/s /( 1/( m^3 * ATOMIC_UNIT_ENERGY ) )
                       step);
                }
            };

        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
