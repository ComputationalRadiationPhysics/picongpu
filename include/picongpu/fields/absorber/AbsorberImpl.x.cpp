/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera, Sergei Bastrakov, Klaus Steiniger
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "picongpu/fields/absorber/AbsorberImpl.hpp"

#include "picongpu/fields/absorber/exponential/Exponential.hpp"
#include "picongpu/fields/absorber/none/None.hpp"
#include "picongpu/fields/absorber/pml/Pml.hpp"
#include "picongpu/particles/filter/filter.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace picongpu
{
    namespace fields
    {
        namespace absorber
        {
            AbsorberImpl::AbsorberImpl(Kind const kind, MappingDesc const cellDescription)
                : Absorber(kind)
                , cellDescription(cellDescription)
            {
            }

            AbsorberImpl& AbsorberImpl::getImpl(MappingDesc const cellDescription)
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

            exponential::ExponentialImpl& AbsorberImpl::asExponentialImpl()
            {
                auto* result = dynamic_cast<exponential::ExponentialImpl*>(this);
                if(!result)
                    throw std::runtime_error("Invalid conversion of absorber to ExponentialImpl");
                return *result;
            }

            pml::PmlImpl& AbsorberImpl::asPmlImpl()
            {
                auto* result = dynamic_cast<pml::PmlImpl*>(this);
                if(!result)
                    throw std::runtime_error("Invalid conversion of absorber to PmlImpl");
                return *result;
            }

            std::unique_ptr<Absorber> AbsorberFactory::make() const
            {
                if(!isInitialized)
                    throw std::runtime_error("Absorber factory used before being initialized");
                auto const instance = Absorber{kind};
                return std::make_unique<Absorber>(instance);
            }

            // This implementation has to go to a .tpp file as it requires definitions of Pml and ExponentialDamping
            std::unique_ptr<AbsorberImpl> AbsorberFactory::makeImpl(MappingDesc const cellDescription) const
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
