/* Copyright 2013-2019 Axel Huebl, Felix Schmitt, Rene Widera
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
#include "picongpu/plugins/hdf5/HDF5Writer.def"
#include "picongpu/traits/PICToSplash.hpp"
#include "picongpu/traits/PICToOpenPMD.hpp"
#include <pmacc/traits/GetComponentsType.hpp>
#include <pmacc/traits/GetNComponents.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/assert.hpp>


namespace picongpu
{

namespace hdf5
{
using namespace pmacc;

using namespace splash;

/** Load attribute of a species from HDF5 checkpoint file
 *
 * @tparam T_Identifier identifier of species attribute
 */
template< typename T_Identifier>
struct LoadParticleAttributesFromHDF5
{

    /** read attributes from hdf5 file
     *
     * @param params thread params with domainwriter, ...
     * @param frame frame with all particles
     * @param subGroup path to the group in the hdf5 file
     * @param particlesOffset read offset in the attribute array
     * @param elements number of elements which should be read the attribute array
     */
    template<typename FrameType>
    HINLINE void operator()(
                            ThreadParams* params,
                            FrameType& frame,
                            const std::string subGroup,
                            const uint64_t particlesOffset,
                            const uint64_t elements)
    {

        typedef T_Identifier Identifier;
        typedef typename pmacc::traits::Resolve<Identifier>::type::type ValueType;
        const uint32_t components = GetNComponents<ValueType>::value;
        typedef typename GetComponentsType<ValueType>::type ComponentType;
        typedef typename PICToSplash<ComponentType>::type SplashType;

        log<picLog::INPUT_OUTPUT > ("HDF5:  ( begin ) load species attribute: %1%") % Identifier::getName();

        const std::string name_lookup[] = {"x", "y", "z"};

        ComponentType* tmpArray = nullptr;
        if( elements > 0 )
            tmpArray = new ComponentType[elements];

        ParallelDomainCollector* dataCollector = params->dataCollector;

        // avoid deadlock between not finished pmacc tasks and mpi calls in splash/HDF5
        __getTransactionEvent().waitForFinished();

        for (uint32_t d = 0; d < components; d++)
        {
            OpenPMDName<T_Identifier> openPMDName;
            std::stringstream datasetName;
            datasetName << subGroup << "/" << openPMDName();
            if (components > 1)
                datasetName << "/" << name_lookup[d];

            ValueType* dataPtr = frame.getIdentifier(Identifier()).getPointer();
            Dimensions sizeRead(0, 0, 0);
            // read one component from file to temporary array
            dataCollector->read(params->currentStep,
                               Dimensions(elements, 1, 1),
                               Dimensions(particlesOffset, 0, 0),
                               datasetName.str().c_str(),
                               sizeRead,
                               tmpArray
                               );
            PMACC_ASSERT(sizeRead[0] == elements);

            /* copy component from temporary array to array of structs */
            #pragma omp parallel for
            for (size_t i = 0; i < elements; ++i)
            {
                ComponentType& ref = ((ComponentType*) dataPtr)[i * components + d];
                ref = tmpArray[i];
            }
        }
        __deleteArray(tmpArray);

        log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) load species attribute: %1%") %
            Identifier::getName();
    }

};

} //namspace hdf5

} //namespace picongpu

