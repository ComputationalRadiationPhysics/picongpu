/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Alexander Debus
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

#if(ENABLE_ADIOS == 1)
#    include <adios.h>

#    include "picongpu/simulation_defines.hpp"
#    include <boost/mpl/if.hpp>
#    include <boost/type_traits.hpp>

namespace picongpu
{
    namespace traits
    {
        template<>
        struct PICToAdios<bool>
        {
            ADIOS_DATATYPES type;

            PICToAdios() : type(adios_unsigned_byte)
            {
            }

            PMACC_STATIC_ASSERT_MSG(sizeof(bool) == 1, ADIOS_Plugin__Can_not_find_a_one_byte_representation_of_bool);
        };

        template<>
        struct PICToAdios<int16_t>
        {
            ADIOS_DATATYPES type;

            PICToAdios() : type(adios_short)
            {
            }
        };

        template<>
        struct PICToAdios<uint16_t>
        {
            ADIOS_DATATYPES type;

            PICToAdios() : type(adios_unsigned_short)
            {
            }
        };

        template<>
        struct PICToAdios<int32_t>
        {
            ADIOS_DATATYPES type;

            PICToAdios() : type(adios_integer)
            {
            }
        };

        template<>
        struct PICToAdios<uint32_t>
        {
            ADIOS_DATATYPES type;

            PICToAdios() : type(adios_unsigned_integer)
            {
            }
        };

        template<>
        struct PICToAdios<int64_t>
        {
            ADIOS_DATATYPES type;

            PICToAdios() : type(adios_long)
            {
            }
        };

        template<>
        struct PICToAdios<uint64_t>
        {
            ADIOS_DATATYPES type;

            PICToAdios() : type(adios_unsigned_long)
            {
            }
        };

        /** Specialization for uint64_cu.
         *  If uint64_cu happens to be the same as uint64_t we use an unused dummy type
         *  to avoid duplicate specialization
         */
        struct uint64_cu_unused_adios;
        template<>
        struct PICToAdios<typename bmpl::if_<
            typename bmpl::
                or_<boost::is_same<uint64_t, uint64_cu>, bmpl::bool_<sizeof(uint64_cu) != sizeof(uint64_t)>>::type,
            uint64_cu_unused_adios,
            uint64_cu>::type> : public PICToAdios<uint64_t>
        {
        };

        template<>
        struct PICToAdios<float_32>
        {
            ADIOS_DATATYPES type;

            PICToAdios() : type(adios_real)
            {
            }
        };

        template<>
        struct PICToAdios<float_64>
        {
            ADIOS_DATATYPES type;

            PICToAdios() : type(adios_double)
            {
            }
        };

    } // namespace traits

} // namespace picongpu

#endif // (ENABLE_ADIOS==1)
