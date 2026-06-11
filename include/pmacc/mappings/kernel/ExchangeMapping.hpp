/*
 * SPDX-FileCopyrightText: Felix Schmitt, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/mappings/kernel/ExchangeMappingMethods.hpp"
#include "pmacc/mappings/kernel/MapperConcept.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /** Mapping from block indices to supercells in the given exchange area for alpaka kernels
     *
     * Adheres to the MapperConcept.
     *
     * Allows mapping thread/block indices to a specific region in a DataSpace
     * defined by a valid ExchangeType combination.
     *
     * @tparam areaType are to map to
     * @tparam baseClass base class for mapping, should be MappingDescription
     */
    template<uint32_t areaType, class baseClass>
    class ExchangeMapping;

    template<uint32_t areaType, template<unsigned, class> class baseClass, unsigned DIM, class SuperCellSize_>
    class ExchangeMapping<areaType, baseClass<DIM, SuperCellSize_>> : public baseClass<DIM, SuperCellSize_>
    {
    private:
        uint32_t exchangeType;

    public:
        using BaseClass = baseClass<DIM, SuperCellSize_>;

        enum
        {
            Dim = BaseClass::Dim
        };

        using SuperCellSize = typename BaseClass::SuperCellSize;

        /**
         * Constructor.
         *
         * @param base object of base class baseClass (see template parameters)
         * @param exchangeType exchange type for mapping
         */
        HINLINE ExchangeMapping(BaseClass base, uint32_t exchangeType) : BaseClass(base), exchangeType(exchangeType)
        {
        }

        ExchangeMapping(ExchangeMapping const&) = default;

        /*get exchange type
         *@return exchange of this object
         */
        HDINLINE uint32_t getExchangeType() const
        {
            return exchangeType;
        }

        /** Generate grid dimension information for alpaka kernel calls
         *
         * A kernel using this mapping must use exacly the returned number of blocks
         *
         * @return number of blocks in a grid
         */
        HINLINE DataSpace<DIM> getGridDim() const
        {
            return ExchangeMappingMethods<areaType, DIM>::getGridDim(*this, exchangeType);
        }

        /** Return index of a supercell to be processed by the given alpaka block
         *
         * @tparam T_origin Which origin (CORE/BORDER/GUARD) to return supercell index relative to (default: GUARD)
         * @param blockIdx alpaka block index
         * @return mapped SuperCell index including guards
         */
        template<uint32_t T_origin = GUARD>
        HDINLINE DataSpace<DIM> getSuperCellIndex(DataSpace<DIM> const& blockIdx) const
        {
            auto result = ExchangeMappingMethods<areaType, DIM>::getSuperCellIndex(*this, blockIdx, exchangeType);
            if constexpr(T_origin == CORE)
            {
                result = result - 2 * this->getGuardingSuperCells();
            }
            if constexpr(T_origin == BORDER)
            {
                result = result - this->getGuardingSuperCells();
            }
            return result;
        }
    };

} // namespace pmacc
