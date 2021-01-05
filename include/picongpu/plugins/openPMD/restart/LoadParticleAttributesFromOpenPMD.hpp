/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Rene Widera, Franz Poeschel
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


#include "picongpu/plugins/openPMD/openPMDWriter.def"
#include "picongpu/simulation_defines.hpp"
#include "picongpu/traits/PICToOpenPMD.hpp"

#include <pmacc/assert.hpp>
#include <pmacc/traits/GetComponentsType.hpp>
#include <pmacc/traits/GetNComponents.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <openPMD/openPMD.hpp>

#include <memory>

namespace picongpu
{
    namespace openPMD
    {
        using namespace pmacc;

        /** Load attribute of a species from openPMD checkpoint storage
         *
         * @tparam T_Identifier identifier of species attribute
         */
        template<typename T_Identifier>
        struct LoadParticleAttributesFromOpenPMD
        {
            /** read attributes from openPMD file
             *
             * @param params thread params
             * @param frame frame with all particles
             * @param particleSpecies the openpmd representation of the species
             * @param particlesOffset read offset in the attribute array
             * @param elements number of elements which should be read the attribute
             * array
             */
            template<typename FrameType>
            HINLINE void operator()(
                ThreadParams* params,
                FrameType& frame,
                ::openPMD::ParticleSpecies particleSpecies,
                const uint64_t particlesOffset,
                const uint64_t elements)
            {
                using Identifier = T_Identifier;
                using ValueType = typename pmacc::traits::Resolve<Identifier>::type::type;
                const uint32_t components = GetNComponents<ValueType>::value;
                using ComponentType = typename GetComponentsType<ValueType>::type;
                OpenPMDName<Identifier> openPMDName;

                log<picLog::INPUT_OUTPUT>("openPMD: ( begin ) load species attribute: %1%") % openPMDName();

                const std::string name_lookup[] = {"x", "y", "z"};

                std::shared_ptr<ComponentType> loadBfr;
                if(elements > 0)
                {
                    loadBfr = std::shared_ptr<ComponentType>{new ComponentType[elements], [](ComponentType* ptr) {
                                                                 delete[] ptr;
                                                             }};
                }

                for(uint32_t n = 0; n < components; ++n)
                {
                    ::openPMD::Record record = particleSpecies[openPMDName()];
                    ::openPMD::RecordComponent rc
                        = components > 1 ? record[name_lookup[n]] : record[::openPMD::RecordComponent::SCALAR];

                    ValueType* dataPtr = frame.getIdentifier(Identifier()).getPointer();

                    if(elements > 0)
                    {
                        // avoid deadlock between not finished pmacc tasks and mpi
                        // calls in openPMD
                        __getTransactionEvent().waitForFinished();
                        rc.loadChunk<ComponentType>(
                            loadBfr,
                            ::openPMD::Offset{particlesOffset},
                            ::openPMD::Extent{elements});
                    }

                    /** start a blocking read of all scheduled variables
                     *  (this is collective call in many methods of openPMD
                     * backends)
                     */
                    params->openPMDSeries->flush();

                    uint64_t globalNumElements = 1;
                    for(auto ext : rc.getExtent())
                    {
                        globalNumElements *= ext;
                    }

                    log<picLog::INPUT_OUTPUT>("openPMD:  Did read %1% local of %2% global elements for "
                                              "%3%")
                        % elements % globalNumElements % openPMDName();

/* copy component from temporary array to array of structs */
#pragma omp parallel for simd
                    for(size_t i = 0; i < elements; ++i)
                    {
                        ComponentType* ref = &reinterpret_cast<ComponentType*>(dataPtr)[i * components + n];
                        *ref = loadBfr.get()[i];
                    }
                }

                log<picLog::INPUT_OUTPUT>("openPMD:  ( end ) load species attribute: %1%") % openPMDName();
            }
        };

    } /* namespace openPMD */
} /* namespace picongpu */
