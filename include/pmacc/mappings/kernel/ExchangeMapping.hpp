/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera, Sergei Bastrakov
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/mappings/kernel/ExchangeMappingMethods.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    template<uint32_t areaType, class baseClass>
    class ExchangeMapping;

    /**
     * Allows mapping thread/block indices to a specific region in a DataSpace
     * defined by a valid ExchangeType combination.
     *
     * @tparam areaType are to map to
     * @tparam baseClass base class for mapping, should be MappingDescription
     */
    template<uint32_t areaType, template<unsigned, class> class baseClass, unsigned DIM, class SuperCellSize_>
    class ExchangeMapping<areaType, baseClass<DIM, SuperCellSize_>> : public baseClass<DIM, SuperCellSize_>
    {
    private:
        uint32_t exchangeType;

    public:
        typedef baseClass<DIM, SuperCellSize_> BaseClass;

        enum
        {
            Dim = BaseClass::Dim
        };


        typedef typename BaseClass::SuperCellSize SuperCellSize;

        /**
         * Constructor.
         *
         * @param base object of base class baseClass (see template parameters)
         * @param exchangeType exchange type for mapping
         */
        HINLINE ExchangeMapping(BaseClass base, uint32_t exchangeType) : BaseClass(base), exchangeType(exchangeType)
        {
        }

        /*get exchange type
         *@return exchange of this object
         */
        HDINLINE uint32_t getExchangeType() const
        {
            return exchangeType;
        }

        /**
         * Generate grid dimension information for kernel calls
         *
         * @return size of the grid
         */
        HINLINE DataSpace<DIM> getGridDim() const
        {
            return ExchangeMappingMethods<areaType, DIM>::getGridDim(*this, exchangeType);
        }

        /**
         * Returns index of current logical block
         *
         * @param realSuperCellIdx current SuperCell index (block index)
         * @return mapped SuperCell index
         */
        HDINLINE DataSpace<DIM> getSuperCellIndex(const DataSpace<DIM>& realSuperCellIdx) const
        {
            return ExchangeMappingMethods<areaType, DIM>::getBlockIndex(*this, realSuperCellIdx, exchangeType);
        }
    };

    /** Construct an exchange mapper instance for the given standard area and description
     *
     * Adheres to the AreaMapperFactoryConcept.
     *
     * @tparam T_area area, a value from type::AreaType or a sum of such values
     */
    template<uint32_t T_area>
    class ExchangeMapperFactory
    {
    public:
        ExchangeMapperFactory(uint32_t exchangeType) : exchangeType(exchangeType)
        {
        }

        /** Construct an exchange mapper object
         *
         * @tparam T_MappingDescription mapping description type
         *
         * @param mappingDescription mapping description
         *
         * @return an object adhering to the ExchangeMapping concept
         */
        template<typename T_MappingDescription>
        HINLINE auto operator()(T_MappingDescription mappingDescription) const
        {
            return ExchangeMapping<T_area, T_MappingDescription>{mappingDescription, exchangeType};
        }

    private:
        //! Exchange type for mapping
        uint32_t exchangeType;
    };

    /** Construct an exchange mapper instance for the given area and description
     *
     * @tparam T_area area, a value from type::AreaType or a sum of such values
     * @tparam T_MappingDescription mapping description type
     *
     * @param mappingDescription mapping description
     * @param exchangeType exchange type for mapping
     */
    template<uint32_t T_area, typename T_MappingDescription>
    HINLINE auto makeExchangeMapper(T_MappingDescription mappingDescription, uint32_t exchangeType)
    {
        return ExchangeMapperFactory<T_area>{exchangeType}(mappingDescription);
    }

} // namespace pmacc
