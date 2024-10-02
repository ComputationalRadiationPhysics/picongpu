/* Copyright 2013-2023 Axel Huebl, Rene Widera, Sergei Bastrakov, Klaus Steiniger
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

#include "picongpu/defines.hpp"
#include "picongpu/fields/absorber/Absorber.hpp"

#include <cstdint>
#include <memory>
#include <string>


namespace picongpu
{
    namespace fields
    {
        namespace absorber
        {
            /** Base class for implementation of absorbers
             *
             * It is currently in an intermediate state due to transition to run-time absorber selection and
             * unification of field solvers.
             * So the base class interface does not offer any common interface but type casts.
             *
             * The reason it is separated from the Absorber class is to better manage lifetime.
             *
             * For clients the class behaves in a singleton-like fashion, with getImpl() for instance access.
             */
            class AbsorberImpl : public Absorber
            {
            public:
                /** Create absorber implementation instance
                 *
                 * @param cellDescription mapping for kernels
                 */
                AbsorberImpl(Kind kind, MappingDesc cellDescription);

                //! Destructor
                ~AbsorberImpl() override = default;

                /** Get absorber implementation instance
                 *
                 * Must always be called with same cellDescription, this is checked inside.
                 * This is a bit awkward and ultimately caused by absorbers being stuck in intermediate state
                 * between compile- and runtime polymorphism.
                 *
                 * @param cellDescription mapping for kernels
                 */
                static AbsorberImpl& getImpl(MappingDesc cellDescription);

                /** Interpret this as ExponentialImpl instance
                 *
                 * @return reference to this object if conversion is valid,
                 *         throws otherwise
                 */
                exponential::ExponentialImpl& asExponentialImpl();

                /** Interpret this as PmlImpl instance
                 *
                 * @return reference to this object if conversion is valid,
                 *         throws otherwise
                 */
                pml::PmlImpl& asPmlImpl();

            protected:
                //! Mapping description for kernels
                MappingDesc cellDescription;
            };

            /** Singletone factory class to construct absorber instances according to the preset kind
             *
             * This class is intended to be used only during initialization of the simulation and by Absorber itself.
             */
            class AbsorberFactory
            {
            public:
                //! Get instance of the factory
                static AbsorberFactory& get()
                {
                    static AbsorberFactory instance;
                    return instance;
                }

                //! Make an absorber instance
                std::unique_ptr<Absorber> make() const;

                /** Make an absorber implementation instance
                 *
                 * @param cellDescription mapping for kernels
                 */
                std::unique_ptr<AbsorberImpl> makeImpl(MappingDesc cellDescription) const;

                /** Set absorber kind to be made
                 *
                 * @param newKind new absorber kind
                 */
                void setKind(Absorber::Kind newKind)
                {
                    kind = newKind;
                    isInitialized = true;
                }

            private:
                Absorber::Kind kind;
                bool isInitialized = false;
            };

        } // namespace absorber
    } // namespace fields
} // namespace picongpu
