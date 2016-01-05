/**
 * Copyright 2013-2016 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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


#include "pmacc_types.hpp"
#include "simulation_types.hpp"
#include "plugins/hdf5/HDF5Writer.def"
#include "traits/PICToSplash.hpp"
#include "traits/PICToOpenPMD.hpp"
#include "traits/GetComponentsType.hpp"
#include "traits/GetNComponents.hpp"
#include "traits/Resolve.hpp"

namespace picongpu
{

namespace hdf5
{
using namespace PMacc;

using namespace splash;


/** write attribute of a particle to hdf5 file
 *
 * @tparam T_Identifier identifier of a particle attribute
 */
template< typename T_Identifier>
struct ParticleAttribute
{
    /** write attribute to hdf5 file
     *
     * @param params wrapped params with domainwriter, ...
     * @param frame frame with all particles
     * @param prefix a name prefix for hdf5 attribute (is combined to: prefix_nameOfAttribute)
     * @param simOffset offset from window origin of the domain
     * @param localSize local domain size
     */
    template<typename FrameType>
    HINLINE void operator()(
                            ThreadParams* params,
                            FrameType& frame,
                            const std::string subGroup,
                            const size_t elements)
    {

        typedef T_Identifier Identifier;
        typedef typename PMacc::traits::Resolve<Identifier>::type::type ValueType;
        const uint32_t components = GetNComponents<ValueType>::value;
        typedef typename GetComponentsType<ValueType>::type ComponentType;
        typedef typename PICToSplash<ComponentType>::type SplashType;
        typedef typename PICToSplash<float_X>::type SplashFloatXType;

        const ThreadParams *threadParams = params;

        log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) write species attribute: %1%") % Identifier::getName();

        SplashType splashType;
        ColTypeDouble ctDouble;
        SplashFloatXType splashFloatXType;

        OpenPMDName<T_Identifier> openPMDName;
        std::string recordName = subGroup + std::string("/") + openPMDName();

        const std::string name_lookup[] = {"x", "y", "z"};

        OpenPMDUnit<T_Identifier> openPMDUnit;
        std::vector<float_64> unit = openPMDUnit();
        OpenPMDUnitDimension<T_Identifier> openPMDUnitDimension;
        std::vector<float_64> unitDimension = openPMDUnitDimension();

        assert(unit.size() == components); // unitSI for each component
        assert(unitDimension.size() == 7); // seven openPMD base units

        /* globalSlideOffset due to gpu slides between origin at time step 0
         * and origin at current time step
         * ATTENTION: splash offset are globalSlideOffset + picongpu offsets
         */
        DataSpace<simDim> globalSlideOffset;
        const PMacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
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
            datasetName << recordName;
            if (components > 1)
                datasetName << "/" << name_lookup[d];

            ValueType* dataPtr = frame.getIdentifier(Identifier()).getPointer();
            #pragma omp parallel for
            for (size_t i = 0; i < elements; ++i)
            {
                tmpArray[i] = ((ComponentValueType*)dataPtr)[i * components + d];
            }

            threadParams->dataCollector->writeDomain(threadParams->currentStep,
                                                     splashType,
                                                     1u,
                                                     splash::Selection(Dimensions(elements, 1, 1)),
                                                     datasetName.str().c_str(),
                                                     splash::Domain(
                                                            splashDomainOffset,
                                                            splashDomainSize
                                                     ),
                                                     splash::Domain(
                                                            splashGlobalDomainOffset,
                                                            splashGlobalDomainSize
                                                     ),
                                                     DomainCollector::PolyType,
                                                     tmpArray);

            threadParams->dataCollector->writeAttribute(threadParams->currentStep,
                                                        ctDouble, datasetName.str().c_str(),
                                                        "unitSI", &(unit.at(d)));

        }
        __deleteArray(tmpArray);


        params->dataCollector->writeAttribute(params->currentStep,
                                              ctDouble, recordName.c_str(),
                                              "unitDimension",
                                              1u, Dimensions(7,0,0),
                                              &(*unitDimension.begin()));

        /** \todo check if always correct at this point, depends on attribute
         *        and MW-solver/pusher implementation */
        const float_X timeOffset = 0.0;
        threadParams->dataCollector->writeAttribute(params->currentStep,
                                                    splashFloatXType, recordName.c_str(),
                                                    "timeOffset", &timeOffset);

        log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) write species attribute: %1%") %
            Identifier::getName();
    }

};

} //namspace hdf5

} //namespace picongpu

