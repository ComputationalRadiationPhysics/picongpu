/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera, Sergei Bastrakov, Klaus Steiniger
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/traits/GetStringProperties.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace picongpu::fields::absorber
{
    //! Thickness of the absorbing layer
    class Thickness
    {
    public:
        //! Create a zero thickness
        Thickness()
        {
            for(uint32_t axis = 0u; axis < 3u; axis++)
                for(uint32_t direction = 0u; direction < 2u; direction++)
                    (*this)(axis, direction) = 0u;
        }

        /** Get thickness for the given boundary
         *
         * @param axis axis, 0 = x, 1 = y, 2 = z
         * @param direction direction, 0 = negative (min coordinate),
         *                  1 = positive (max coordinate)
         */
        uint32_t operator()(uint32_t const axis, uint32_t const direction) const
        {
            return numCells[axis][direction];
        }

        /** Get reference to thickness for the given boundary
         *
         * @param axis axis, 0 = x, 1 = y, 2 = z
         * @param direction direction, 0 = negative (min coordinate),
         *                  1 = positive (max coordinate)
         */
        uint32_t& operator()(uint32_t const axis, uint32_t const direction)
        {
            return numCells[axis][direction];
        }

        //! Get thickness for the negative border, at the local domain sides minimum in coordinates
        pmacc::DataSpace<simDim> getNegativeBorder() const
        {
            pmacc::DataSpace<simDim> result;
            for(uint32_t axis = 0u; axis < simDim; axis++)
                result[axis] = (*this)(axis, 0);
            return result;
        }

        //! Get thickness for the positive border, at the local domain sides maximum in coordinates
        pmacc::DataSpace<simDim> getPositiveBorder() const
        {
            pmacc::DataSpace<simDim> result;
            for(uint32_t axis = 0u; axis < simDim; axis++)
                result[axis] = (*this)(axis, 1);
            return result;
        }

    private:
        /** Number of absorber cells along each boundary
         *
         * First index: 0 = x, 1 = y, 2 = z.
         * Second index: 0 = negative (min coordinate), 1 = positive (max coordinate).
         */
        uint32_t numCells[3][2];
    };
} // namespace picongpu::fields::absorber
