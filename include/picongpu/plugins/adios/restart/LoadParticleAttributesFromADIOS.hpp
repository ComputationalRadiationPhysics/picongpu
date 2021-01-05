/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Rene Widera
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
#include "picongpu/plugins/adios/ADIOSWriter.def"
#include "picongpu/plugins/misc/ComponentNames.hpp"
#include "picongpu/traits/PICToOpenPMD.hpp"
#include <pmacc/traits/GetComponentsType.hpp>
#include <pmacc/traits/GetNComponents.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/assert.hpp>


namespace picongpu
{
    namespace adios
    {
        using namespace pmacc;

        /** Load attribute of a species from ADIOS checkpoint file
         *
         * @tparam T_Identifier identifier of species attribute
         */
        template<typename T_Identifier>
        struct LoadParticleAttributesFromADIOS
        {
            /** read attributes from ADIOS file
             *
             * @param params thread params with ADIOS_FILE, ...
             * @param frame frame with all particles
             * @param particlePath path to the group in the ADIOS file
             * @param particlesOffset read offset in the attribute array
             * @param elements number of elements which should be read the attribute array
             */
            template<typename FrameType>
            HINLINE void operator()(
                ThreadParams* params,
                FrameType& frame,
                const std::string particlePath,
                const uint64_t particlesOffset,
                const uint64_t elements)
            {
                typedef T_Identifier Identifier;
                typedef typename pmacc::traits::Resolve<Identifier>::type::type ValueType;
                const uint32_t components = GetNComponents<ValueType>::value;
                typedef typename GetComponentsType<ValueType>::type ComponentType;

                log<picLog::INPUT_OUTPUT>("ADIOS: ( begin ) load species attribute: %1%") % Identifier::getName();

                const auto componentNames = plugins::misc::getComponentNames(components);

                ComponentType* tmpArray = nullptr;
                if(elements > 0)
                    tmpArray = new ComponentType[elements];

                // dev assert!
                if(elements > 0)
                    PMACC_ASSERT(tmpArray);

                for(uint32_t n = 0; n < components; ++n)
                {
                    OpenPMDName<T_Identifier> openPMDName;
                    std::stringstream datasetName;
                    datasetName << particlePath << openPMDName();
                    if(components > 1)
                        datasetName << "/" << componentNames[n];

                    ValueType* dataPtr = frame.getIdentifier(Identifier()).getPointer();

                    ADIOS_VARINFO* varInfo = adios_inq_var(params->fp, datasetName.str().c_str());
                    // it's possible to aquire the local block with that call again and
                    // the local elements to-be-read, but the block-ID must be known (MPI rank?)
                    // ADIOS_CMD(adios_inq_var_blockinfo( params->fp, varInfo ));

                    ADIOS_SELECTION* sel = adios_selection_boundingbox(1, &particlesOffset, &elements);

                    /** Note: adios_schedule_read is not a collective call in any
                     *        ADIOS method and can therefore be skipped for empty reads */
                    if(elements > 0)
                    {
                        // avoid deadlock between not finished pmacc tasks and mpi calls in adios
                        __getTransactionEvent().waitForFinished();
                        ADIOS_CMD(adios_schedule_read(
                            params->fp,
                            sel,
                            datasetName.str().c_str(),
                            0, /* from_step (not used in streams) */
                            1, /* nsteps to read (must be 1 for stream) */
                            (void*) tmpArray));
                    }

                    /** start a blocking read of all scheduled variables
                     *  (this is collective call in many ADIOS methods) */
                    ADIOS_CMD(adios_perform_reads(params->fp, 1));

                    log<picLog::INPUT_OUTPUT>("ADIOS:  Did read %1% local of %2% global elements for %3%") % elements
                        % varInfo->dims[0] % datasetName.str();

/* copy component from temporary array to array of structs */
#pragma omp parallel for
                    for(size_t i = 0; i < elements; ++i)
                    {
                        ComponentType& ref = ((ComponentType*) dataPtr)[i * components + n];
                        ref = tmpArray[i];
                    }

                    adios_selection_delete(sel);
                    adios_free_varinfo(varInfo);
                }
                __deleteArray(tmpArray);

                log<picLog::INPUT_OUTPUT>("ADIOS:  ( end ) load species attribute: %1%") % Identifier::getName();
            }
        };

    } /* namespace adios */

} /* namespace picongpu */
