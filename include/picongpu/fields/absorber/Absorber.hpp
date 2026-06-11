/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera, Sergei Bastrakov, Klaus Steiniger
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/absorber/Thickness.hpp"

#include <pmacc/traits/GetStringProperties.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace picongpu
{
    namespace fields
    {
        namespace absorber
        {
            /** Singleton for field absorber
             *
             * Provides run-time utilities to get thickness and string properties.
             * Does not provide absorption implmenetation itself, that is done by AbsorberImpl.
             */
            class Absorber
            {
            public:
                /** Supported absorber kinds, same for all absorbing boundaries
                 *
                 * Exponential - exponential damping absorber.
                 * None - all boundaries are periodic, no absorber.
                 * Pml - perfectly matched layer absorber.
                 */
                enum class Kind
                {
                    Exponential,
                    None,
                    Pml
                };

                //! Destructor needs to be public due to internal use of std::unique_ptr
                virtual ~Absorber() = default;

                //! Get absorber instance
                static Absorber& get();

                //! Absorber kind used in the simulation
                Kind getKind() const;

                /** Get absorber thickness in number of cells for the global domain
                 *
                 * This function takes into account which boundaries are periodic and absorbing.
                 */
                Thickness getGlobalThickness() const;

                /** Get absorber thickness in number of cells for the current local domain
                 *
                 * This function takes into account the current domain decomposition and
                 * which boundaries are periodic and absorbing.
                 *
                 * Note that unlike getGlobalThickness() result which does not change
                 * throughout the simulation, the local thickness can change.
                 * Thus, the result of this function should not be reused on another time step,
                 * but rather the function called again.
                 */
                Thickness getLocalThickness() const;

                //! Get string properties
                static pmacc::traits::StringProperty getStringProperties();

            protected:
                /** Number of absorber cells along each boundary
                 *
                 * Stores the global absorber thickness along each boundary.
                 * Note that in case of periodic
                 * boundaries the corresponding values will be ignored.
                 *
                 * Is uniform for both PML and exponential damping absorbers.
                 * First index: 0 = x, 1 = y, 2 = z.
                 * Second index: 0 = negative (min coordinate), 1 = positive (max coordinate).
                 */
                uint32_t numCells[3][2];

                //! Absorber kind
                Kind kind;

                //! Text name for string properties
                std::string name;

                //! Create absorber with the given kind
                Absorber(Kind kind);

                friend class AbsorberFactory;
            };

            // Forward declaration for AbsorberImpl::asExponentialImpl()
            namespace exponential
            {
                class ExponentialImpl;
            } // namespace exponential

            // Forward declaration for AbsorberImpl::asPmlImpl()
            namespace pml
            {
                class PmlImpl;
            } // namespace pml


        } // namespace absorber
    } // namespace fields
} // namespace picongpu
