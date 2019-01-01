/* Copyright 2013-2019 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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


/** write attribute of a particle to hdf5 file
 *
 * @tparam T_Identifier identifier of a particle record
 */
template< typename T_Identifier>
struct ParticleAttribute
{
    /** write attribute to hdf5 file
     *
     * @param params wrapped thread params such as domainwriter, ...
     * @param frame frame with all particles
     * @param speciesPath path for the current species (of FrameType)
     * @param elements number of particles in this patch
     * @param elementsOffset number of particles in this patch
     * @param numParticlesGlobal number of particles globally
     */
    template<typename FrameType>
    HINLINE void operator()(
                            ThreadParams* params,
                            FrameType& frame,
                            const std::string speciesPath,
                            const uint64_t elements,
                            const uint64_t elementsOffset,
                            const uint64_t numParticlesGlobal
    )
    {

        typedef T_Identifier Identifier;
        typedef typename pmacc::traits::Resolve<Identifier>::type::type ValueType;
        const uint32_t components = GetNComponents<ValueType>::value;
        typedef typename GetComponentsType<ValueType>::type ComponentType;
        typedef typename PICToSplash<ComponentType>::type SplashType;
        typedef typename PICToSplash<float_X>::type SplashFloatXType;

        const ThreadParams *threadParams = params;

        log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) write species attribute: %1%") % Identifier::getName();

        SplashType splashType;
        ColTypeDouble ctDouble;
        ColTypeUInt32 ctUInt32;
        SplashFloatXType splashFloatXType;

        OpenPMDName<T_Identifier> openPMDName;
        const std::string recordPath( speciesPath + std::string("/") + openPMDName() );

        const std::string name_lookup[] = {"x", "y", "z"};

        // get the SI scaling, dimensionality and weighting of the attribute
        OpenPMDUnit<T_Identifier> openPMDUnit;
        std::vector<float_64> unit = openPMDUnit();
        OpenPMDUnitDimension<T_Identifier> openPMDUnitDimension;
        std::vector<float_64> unitDimension = openPMDUnitDimension();
        const bool macroWeightedBool = MacroWeighted<T_Identifier>::get();
        const uint32_t macroWeighted = (macroWeightedBool ? 1 : 0);
        const float_64 weightingPower = WeightingPower<T_Identifier>::get();

        PMACC_ASSERT(unit.size() == components); // unitSI for each component
        PMACC_ASSERT(unitDimension.size() == 7); // seven openPMD base units

        /* globalSlideOffset due to gpu slides between origin at time step 0
         * and origin at current time step
         * ATTENTION: splash offset are globalSlideOffset + picongpu offsets
         */
        DataSpace<simDim> globalSlideOffset;
        const pmacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
        const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(threadParams->currentStep);
        globalSlideOffset.y() += numSlides * localDomain.size.y();

        Dimensions splashDomainOffset(0, 0, 0);
        Dimensions splashGlobalDomainOffset(0, 0, 0);

        Dimensions splashDomainSize(1, 1, 1);
        Dimensions splashGlobalDomainSize(1, 1, 1);

        for (uint32_t d = 0; d < simDim; ++d)
        {
            splashDomainOffset[d] = threadParams->window.localDimensions.offset[d] + globalSlideOffset[d];
            splashGlobalDomainOffset[d] = threadParams->window.globalDimensions.offset[d] + globalSlideOffset[d];
            splashGlobalDomainSize[d] = threadParams->window.globalDimensions.size[d];
            splashDomainSize[d] = threadParams->window.localDimensions.size[d];
        }

        typedef typename GetComponentsType<ValueType>::type ComponentValueType;

        ComponentValueType* tmpArray = new ComponentValueType[elements];

        for (uint32_t d = 0; d < components; d++)
        {
            std::stringstream datasetName;
            datasetName << recordPath;
            if (components > 1)
                datasetName << "/" << name_lookup[d];

            ValueType* dataPtr = frame.getIdentifier(Identifier()).getPointer();
            #pragma omp parallel for
            for( uint64_t i = 0; i < elements; ++i )
            {
                tmpArray[i] = ((ComponentValueType*)dataPtr)[i * components + d];
            }

            // avoid deadlock between not finished pmacc tasks and mpi calls in splash/HDF5
            __getTransactionEvent().waitForFinished();
            threadParams->dataCollector->writeDomain(
                threadParams->currentStep,
                /* Dimensions for global collective buffer */
                Dimensions(numParticlesGlobal, 1, 1),
                /* 3D-offset in the globalSize-buffer this process writes to */
                Dimensions(elementsOffset, 1, 1),
                /* Type information for data */
                splashType,
                /* Number of dimensions (1-3) of the buffer */
                1u,
                /* Selection: size in src buffer */
                splash::Selection(
                    Dimensions(elements, 1, 1)
                ),
                /* Name of the dataset */
                datasetName.str().c_str(),
                /* Global domain information */
                splash::Domain(
                    splashGlobalDomainOffset,
                    splashGlobalDomainSize
                ),
                /* Domain type annotation */
                DomainCollector::PolyType,
                /* Buffer with data */
                tmpArray
            );

            threadParams->dataCollector->writeAttribute(
                threadParams->currentStep,
                ctDouble, datasetName.str().c_str(),
                "unitSI", &(unit.at(d)));

        }
        __deleteArray(tmpArray);


        threadParams->dataCollector->writeAttribute(
            params->currentStep,
            ctDouble, recordPath.c_str(),
            "unitDimension",
            1u, Dimensions(7,0,0),
            &(*unitDimension.begin()));

        threadParams->dataCollector->writeAttribute(
            params->currentStep,
            ctUInt32, recordPath.c_str(),
            "macroWeighted",
            &macroWeighted);

        threadParams->dataCollector->writeAttribute(
            params->currentStep,
            ctDouble, recordPath.c_str(),
            "weightingPower",
            &weightingPower);

        /** \todo check if always correct at this point, depends on attribute
         *        and MW-solver/pusher implementation */
        const float_X timeOffset = 0.0;
        threadParams->dataCollector->writeAttribute(params->currentStep,
                                                    splashFloatXType, recordPath.c_str(),
                                                    "timeOffset", &timeOffset);

        log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) write species attribute: %1%") %
            Identifier::getName();
    }

};

} //namspace hdf5

} //namespace picongpu

