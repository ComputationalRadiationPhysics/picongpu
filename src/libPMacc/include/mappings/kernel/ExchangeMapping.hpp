/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef EXCHANGEMAPPING_H
#define	EXCHANGEMAPPING_H

#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "mappings/kernel/ExchangeMappingMethods.hpp"

namespace PMacc
{

    template<uint32_t areaType, class baseClass>
    class ExchangeMapping;

    /**
     * Allows mapping CUDA thread/block indizes to a specific region in a DataSpace
     * defined by a valid ExchangeType combination.
     *
     * @tparam areaType are to map to
     * @tparam baseClass base class for mapping, should be MappingDescription
     */
    template<
    uint32_t areaType,
    template<unsigned, class> class baseClass,
    unsigned DIM,
    class SuperCellSize_
    >
    class ExchangeMapping<areaType, baseClass<DIM, SuperCellSize_> > : public baseClass<DIM, SuperCellSize_>
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
        HINLINE ExchangeMapping(BaseClass base, uint32_t exchangeType) :
        BaseClass(base),
        exchangeType(exchangeType)
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
         * Generates cuda gridDim information for kernel call.
         *
         * @return DataSpace with gridDim information
         */
        HINLINE DataSpace<DIM> getGridDim()
        {
            return this->reduce(ExchangeMappingMethods<areaType, DIM>::getGridDim(*this, exchangeType));
        }

        /**
         * Returns index of current logical block, depending on current cuda block id.
         *
         * @param _blockIdx current cuda block id (blockIdx)
         * @return current logical block index
         */
        DINLINE DataSpace<DIM> getSuperCellIndex(const DataSpace<DIM>& realSuperCellIdx)
        {
            return ExchangeMappingMethods<areaType, DIM>::getBlockIndex(
                                                                        *this,
                                                                        extend(realSuperCellIdx),
                                                                        exchangeType);
        }

    };


} // namespace PMacc



#endif	/* EXCHANGEMAPPING_H */

