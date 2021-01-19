/* Copyright 2013-2020 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
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

// actual call to kernel found here
#include <picongpu/particles/atomicPhysics/CallAtomicPhysics.hpp>

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/Environment.hpp>

#include <cstdint>
#include <stdio.h>


/** @file
 *
 * This file implements the Atomic Physics stage of the PIC-loop.
 *
 * One instance of this class AtomicPhysics is stored as a protected member of the
 * MySimulation class.
 *
 * @private-member:
 * - MappingDesc cellDescription 
 *      Description of cell structure used for PIC-Simulations, ask real programmers
 *       for more information ;)
 *      @todo add pointer to documentation of cell description; 15.12.2020-Brian Marre
 * - <list like> SpeciesWithAtomicPhysics
 *      seqential, list like, of all species with flag _atomicPhysics as defined in
 *      species.param
 * - <functor> callAtomicPhysics
 *      functor for actual atomic physics call for a given species of macroparticles
 *      definded in file <include/picongpu/particles/atomicPhysics/CallAtomicPhysics.hpp>
 *
 * @operator:
 * - ( step ):
 *      calls the callatomicPhysics functor passing the cellDescription and current
 *      time step.
 *      this operator is called once per time step by the simulation main loop
 */

namespace picongpu
{
namespace simulation
{
namespace stage
{

    //! test stage for AtomicPhysics
    // one instance of this class is initialised and it's operator() called for every time step
    class AtomicPhysics
    {
    public:

        AtomicPhysics( MappingDesc const cellDescription ):
            cellDescription( cellDescription )
        {
        }

        void operator( )( uint32_t const step ) const
        {
            // create instance 
            callAtomicPhysics(
                step,
                cellDescription
            );
        }

    private:

        // list of all species of macro particles with flag _atomicPhysics as defined in
        // species.param, is list of types
        using SpeciesWithAtomicPhysics = typename pmacc::particles::traits::FilterByFlag<
            VectorAllSpecies,
            // temporary name
            _atomicPhysics< >
        >::type;

        // kernel to be called for each species
        pmacc::meta::ForEach<
            SpeciesWithAtomicPhysics,
            particles::atomicPhysics::CallAtomicPhysics< bmpl::_1 >
        > callAtomicPhysics;

        //! Mapping for kernels
        MappingDesc cellDescription;

    };

} // namespace stage
} // namespace simulation
} // namespace picongpu
