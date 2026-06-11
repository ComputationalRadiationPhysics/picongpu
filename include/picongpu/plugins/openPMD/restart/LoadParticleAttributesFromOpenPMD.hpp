/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Rene Widera, Franz Poeschel
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#if (ENABLE_OPENPMD == 1)

#    include "picongpu/defines.hpp"
#    include "picongpu/plugins/openPMD/GetComponentsType.hpp"
#    include "picongpu/plugins/openPMD/openPMDWriter.def"
#    include "picongpu/traits/PICToOpenPMD.hpp"

#    include <pmacc/assert.hpp>
#    include <pmacc/traits/GetComponentsType.hpp>
#    include <pmacc/traits/GetNComponents.hpp>
#    include <pmacc/traits/Resolve.hpp>

#    include <memory>

#    include <openPMD/openPMD.hpp>

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
                uint64_t const particlesOffset,
                uint64_t const elements)
            {
                using Identifier = T_Identifier;
                using ValueType = typename pmacc::traits::Resolve<Identifier>::type::type;
                uint32_t const components = GetNComponents<ValueType>::value;
                using ComponentType = typename GetComponentsType<ValueType>::type;
                picongpu::traits::OpenPMDName<Identifier> openPMDName;

                log<picLog::INPUT_OUTPUT>("openPMD: ( begin ) load species attribute: %1%") % openPMDName();

                std::string const name_lookup[] = {"x", "y", "z"};

                std::shared_ptr<ComponentType> loadBfr;
                if(elements > 0)
                {
                    loadBfr = std::shared_ptr<ComponentType>{
                        new ComponentType[elements],
                        [](ComponentType* ptr) { delete[] ptr; }};
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
                        eventSystem::getTransactionEvent().waitForFinished();
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
#    pragma omp parallel for simd
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

#endif
