/* Copyright 2014-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Franz Poeschel
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/plugins/openPMD/GetComponentsType.hpp"
#include "picongpu/plugins/openPMD/openPMDDimension.hpp"
#include "picongpu/plugins/openPMD/openPMDWriter.def"
#include "picongpu/traits/PICToOpenPMD.tpp"

#include <pmacc/traits/GetComponentsType.hpp>
#include <pmacc/traits/GetNComponents.hpp>
#include <pmacc/traits/Resolve.hpp>

namespace picongpu
{
    namespace openPMD
    {
        using namespace pmacc;

        static const std::string name_lookup[] = {"x", "y", "z"};

        template<typename T_Identifier>
        struct InitParticleAttribute
        {
            /** create attribute metadata in openPMD series
             *
             * @attention This is a MPI_Collective operation and all MPI ranks must participate.
             *
             * @param params wrapped params
             */
            HINLINE void operator()(
                ThreadParams* params,
                ::openPMD::Container<::openPMD::Record>& particleSpecies,
                std::string const& basepath,
                size_t const globalElements)
            {
                using Identifier = T_Identifier;
                using ValueType = typename pmacc::traits::Resolve<Identifier>::type::type;
                const uint32_t components = GetNComponents<ValueType>::value;
                using ComponentType = typename GetComponentsType<ValueType>::type;

                OpenPMDName<T_Identifier> openPMDName;
                ::openPMD::Record record = particleSpecies[openPMDName()];
                std::string baseName = basepath + "/" + openPMDName();
                ::openPMD::Datatype openPMDType = ::openPMD::determineDatatype<ComponentType>();

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

                auto unitMap = convertToUnitDimension(unitDimension);

                record.setUnitDimension(unitMap);
                record.setAttribute("macroWeighted", macroWeighted);
                record.setAttribute("weightingPower", weightingPower);

                /* @todo check if always correct at this point,
                 * depends on attribute and MW-solver/pusher implementation
                 * Note: This attribute is written again in ParticleAttribute::operator() as a workaround for another
                 * bug. When changing this attribute, also change the below one.
                 */
                float_X const timeOffset = 0.0;
                record.setAttribute("timeOffset", timeOffset);

                for(uint32_t d = 0; d < components; d++)
                {
                    ::openPMD::RecordComponent recordComponent
                        = components > 1 ? record[name_lookup[d]] : record[::openPMD::MeshRecordComponent::SCALAR];

                    std::string datasetName = components > 1 ? baseName + "/" + name_lookup[d] : baseName;
                    params->initDataset<DIM1>(recordComponent, openPMDType, {globalElements}, datasetName);

                    if(unit.size() >= (d + 1))
                    {
                        recordComponent.setUnitSI(unit[d]);
                    }
                }
            }
        };


        /** write attribute of a particle to openPMD series
         *
         * @tparam T_Identifier identifier of a particle attribute
         */
        template<typename T_Identifier>
        struct ParticleAttribute
        {
            /** write attribute to openPMD series
             *
             * @param params wrapped params
             * @param elements elements of this attribute
             */
            template<typename FrameType>
            HINLINE void operator()(
                ThreadParams* params,
                FrameType& frame,
                ::openPMD::Container<::openPMD::Record>& particleSpecies,
                std::string const& basepath,
                const size_t elements,
                const size_t globalElements,
                const size_t globalOffset,
                size_t & accumulate_writtenBytes)
            {
                using Identifier = T_Identifier;
                using ValueType = typename pmacc::traits::Resolve<Identifier>::type::type;
                using ComponentType = typename GetComponentsType<ValueType>::type;

                const uint32_t components = GetNComponents<ValueType>::value;
                OpenPMDName<T_Identifier> openPMDName;
                ::openPMD::Record record = particleSpecies[openPMDName()];

                /*
                 * Bug workaround: Ensure that the containing iteration is flushed. Write some attribute again in order
                 * to forcibly mark it dirty.
                 *
                 * First, smaller bug:
                 *     Only on dev branch, introduced by https://github.com/openPMD/openPMD-api/pull/1598,
                 *     fixed by: https://github.com/openPMD/openPMD-api/pull/1615.
                 *
                 * Second, more troublesome bug, needs a long-term fix upstream:
                 *     https://github.com/openPMD/openPMD-api/issues/1616.
                 */
                {
                    float_X const timeOffset = 0.0;
                    record.setAttribute("timeOffset", timeOffset);
                }
                if(elements == 0)
                {
                    // accumulate_writtenBytes += 0;
                    return;
                }

                log<picLog::INPUT_OUTPUT>("openPMD:  (begin) write species attribute: %1%") % Identifier::getName();

                accumulate_writtenBytes += components * elements * sizeof(ComponentType);

                for(uint32_t d = 0; d < components; d++)
                {
                    ::openPMD::RecordComponent recordComponent
                        = components > 1 ? record[name_lookup[d]] : record[::openPMD::MeshRecordComponent::SCALAR];

                    ValueType* dataPtr = frame.getIdentifier(Identifier()).getPointer(); // can be moved up?
                    // ask openPMD to create a buffer for us
                    // in some backends (ADIOS2), this allows avoiding memory copies
                    auto span
                        = recordComponent
                              .storeChunk<ComponentType>(::openPMD::Offset{globalOffset}, ::openPMD::Extent{elements})
                              .currentBuffer();

/* copy strided data from source to temporary buffer */
#pragma omp parallel for simd
                    for(size_t i = 0; i < elements; ++i)
                    {
                        span[i] = reinterpret_cast<ComponentType*>(dataPtr)[d + i * components];
                    }
                }

                log<picLog::INPUT_OUTPUT>("openPMD:  ( end ) write species attribute: %1%") % Identifier::getName();
            }
        };

    } // namespace openPMD
} // namespace picongpu
