/* Copyright 2021-2023 Sergei Bastrakov
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

#include "picongpu/fields/absorber/Absorber.hpp"
#include "picongpu/fields/absorber/exponential/Exponential.hpp"
#include "picongpu/fields/absorber/none/None.hpp"
#include "picongpu/fields/absorber/pml/Pml.hpp"

#include <pmacc/Environment.hpp>

#include <memory>
#include <sstream>
#include <type_traits>


namespace picongpu
{
    namespace fields
    {
        namespace absorber
        {
            inline Absorber::Absorber(Kind const kind) : kind(kind)
            {
                switch(kind)
                {
                case Kind::Exponential:
                    name = std::string{"exponential damping"};
                    break;
                case Kind::None:
                    name = std::string{"none"};
                    break;
                case Kind::Pml:
                    name = std::string{"convolutional PML"};
                    break;
                default:
                    throw std::runtime_error("Unsupported absorber kind requested to be made");
                }
                // Read thickness from absorber.param
                for(uint32_t axis = 0u; axis < simDim; axis++)
                    for(uint32_t direction = 0u; direction < 2u; direction++)
                        numCells[axis][direction] = NUM_CELLS[axis][direction];
            }

            inline Absorber& Absorber::get()
            {
                // Delay initialization till the first call since the factory has its parameters set during runtime
                static std::unique_ptr<Absorber> pInstance = nullptr;
                if(!pInstance)
                {
                    auto& factory = AbsorberFactory::get();
                    pInstance = factory.make();
                }
                return *pInstance;
            }

            inline Absorber::Kind Absorber::getKind() const
            {
                return kind;
            }

            inline Thickness Absorber::getGlobalThickness() const
            {
                Thickness thickness;
                for(uint32_t axis = 0u; axis < 3u; axis++)
                    for(uint32_t direction = 0u; direction < 2u; direction++)
                        thickness(axis, direction) = numCells[axis][direction];
                const DataSpace<DIM3> isPeriodicBoundary
                    = Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();
                for(uint32_t axis = 0u; axis < 3u; axis++)
                    if(isPeriodicBoundary[axis])
                    {
                        thickness(axis, 0) = 0u;
                        thickness(axis, 1) = 0u;
                    }
                return thickness;
            }

            inline Thickness Absorber::getLocalThickness() const
            {
                Thickness thickness = getGlobalThickness();
                auto const numExchanges = NumberOfExchanges<simDim>::value;
                auto const communicationMask = Environment<simDim>::get().GridController().getCommunicationMask();
                for(uint32_t exchange = 1u; exchange < numExchanges; exchange++)
                {
                    /* Here we are only interested in the positive and negative
                     * directions for x, y, z axes and not the "diagonal" ones.
                     * So skip other directions except left, right, top, bottom,
                     * back, front
                     */
                    if(FRONT % exchange != 0)
                        continue;

                    // Transform exchange into a pair of axis and direction
                    uint32_t axis = 0;
                    if(exchange >= BOTTOM && exchange <= TOP)
                        axis = 1;
                    if(exchange >= BACK)
                        axis = 2;
                    uint32_t direction = exchange % 2;

                    // No absorber at the borders between two local domains
                    bool hasNeighbour = communicationMask.isSet(exchange);
                    if(hasNeighbour)
                        thickness(axis, direction) = 0u;
                }
                return thickness;
            }

            inline pmacc::traits::StringProperty Absorber::getStringProperties()
            {
                auto& absorber = get();
                auto const thickness = absorber.getGlobalThickness();
                pmacc::traits::StringProperty propList;
                const DataSpace<DIM3> periodic
                    = Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();

                for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
                {
                    // for each planar direction: left right top bottom back front
                    if(FRONT % i == 0)
                    {
                        const std::string directionName = ExchangeTypeNames()[i];
                        const DataSpace<DIM3> relDir = Mask::getRelativeDirections<DIM3>(i);

                        bool isPeriodic = false;
                        uint32_t axis = 0; // x(0) y(1) z(2)
                        uint32_t axisDir = 0; // negative (0), positive (1)
                        for(uint32_t d = 0; d < simDim; d++)
                        {
                            if(relDir[d] * periodic[d] != 0)
                                isPeriodic = true;
                            if(relDir[d] != 0)
                                axis = d;
                        }
                        if(relDir[axis] > 0)
                            axisDir = 1;

                        std::string boundaryName = "open"; // absorbing boundary
                        if(isPeriodic)
                            boundaryName = "periodic";

                        if(boundaryName == "open")
                        {
                            std::ostringstream boundaryParam;
                            boundaryParam << absorber.name + " over " << thickness(axis, axisDir) << " cells";
                            propList[directionName]["param"] = boundaryParam.str();
                        }
                        else
                        {
                            propList[directionName]["param"] = "none";
                        }

                        propList[directionName]["name"] = boundaryName;
                    }
                }
                return propList;
            }

            inline AbsorberImpl::AbsorberImpl(Kind const kind, MappingDesc const cellDescription)
                : Absorber(kind)
                , cellDescription(cellDescription)
            {
            }

            inline AbsorberImpl& AbsorberImpl::getImpl(MappingDesc const cellDescription)
            {
                // Delay initialization till the first call since the factory has its parameters set during runtime
                static std::unique_ptr<AbsorberImpl> pInstance = nullptr;
                if(!pInstance)
                {
                    auto& factory = AbsorberFactory::get();
                    pInstance = factory.makeImpl(cellDescription);
                }
                else if(pInstance->cellDescription != cellDescription)
                    throw std::runtime_error("AbsorberImpl::getImpl() called with a different mapping description");
                return *pInstance;
            }

            inline exponential::ExponentialImpl& AbsorberImpl::asExponentialImpl()
            {
                auto* result = dynamic_cast<exponential::ExponentialImpl*>(this);
                if(!result)
                    throw std::runtime_error("Invalid conversion of absorber to ExponentialImpl");
                return *result;
            }

            inline pml::PmlImpl& AbsorberImpl::asPmlImpl()
            {
                auto* result = dynamic_cast<pml::PmlImpl*>(this);
                if(!result)
                    throw std::runtime_error("Invalid conversion of absorber to PmlImpl");
                return *result;
            }

            inline std::unique_ptr<Absorber> AbsorberFactory::make() const
            {
                if(!isInitialized)
                    throw std::runtime_error("Absorber factory used before being initialized");
                auto const instance = Absorber{kind};
                return std::make_unique<Absorber>(instance);
            }

            // This implementation has to go to a .tpp file as it requires definitions of Pml and ExponentialDamping
            inline std::unique_ptr<AbsorberImpl> AbsorberFactory::makeImpl(MappingDesc const cellDescription) const
            {
                if(!isInitialized)
                    throw std::runtime_error("Absorber factory used before being initialized");
                switch(kind)
                {
                case Absorber::Kind::Exponential:
                    return std::make_unique<exponential::ExponentialImpl>(cellDescription);
                case Absorber::Kind::None:
                    return std::make_unique<none::NoneImpl>(cellDescription);
                case Absorber::Kind::Pml:
                    return std::make_unique<pml::PmlImpl>(cellDescription);
                default:
                    throw std::runtime_error("Unsupported absorber kind requested to be made");
                }
            }

        } // namespace absorber
    } // namespace fields
} // namespace picongpu
