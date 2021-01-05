/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera, Wolfgang Hoenig,
 *                     Alexander Grund
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

#include "pmacc/types.hpp"

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"

namespace pmacc
{
    /**
     * Mask is used to describe in which directions data must be
     * sent/received or where a grid node has neighbors.
     */
    class Mask
    {
    public:
        /**
         * Constructor.
         *
         * Sets this mask to 0 (nothing).
         */
        Mask() : bitMask(0u)
        {
        }

        /**
         * Constructor.
         *
         * Sets this mask to directions described by ex
         *
         * @param ex directions for this mask
         */
        Mask(ExchangeType ex) : bitMask(1u << ex)
        {
        }

        /**
         * Constructor.
         *
         * Sets this mask to directions described by ex
         *
         * @param ex directions for this mask
         */
        Mask(uint32_t ex) : bitMask(1u << ex)
        {
        }

        /**
         * Destructor.
         */
        virtual ~Mask()
        {
        }

        /**
         * Gives uint32_t value of this mask.
         */
        operator uint32_t() const
        {
            return this->bitMask;
        }

        /**
         * Gives uint32_t value of this mask.
         */
        Mask& operator=(uint32_t other)
        {
            bitMask = other;
            return *this;
        }

        /**
         * Joins two masks.
         *
         * Creates join of this mask and other.
         *
         * @param other Mask with directions to join
         * @return the newly created mask
         */
        Mask operator+(const Mask& other) const
        {
            Mask result;
            result.bitMask = bitMask | other.bitMask;
            return result;
        }

        /**
         * Intersects two masks.
         *
         * Creates intersection of this mask and other.
         *
         * @param other Mask with directions to intersect with
         * @return the newly created mask
         */
        Mask operator&(const Mask& other) const
        {
            Mask result;
            result.bitMask = bitMask & other.bitMask;
            return result;
        }

        /**
         * Returns if ExchangeType direction ex is included in mask direction.
         *
         * examples:
         * Mask(RIGHT).containsExchangeType(RIGHT) == true
         * Mask(RIGHT+FRONT).containsExchangeType(RIGHT) == true
         * Mask(RIGHT)+Mask(FRONT).containsExchangeType(RIGHT) == true
         *
         * @param ex ExchangeType to query
         * @return true if ex is included in mask directions, false otherwise
         */
        HDINLINE bool containsExchangeType(uint32_t ex) const
        {
            for(uint32_t i = 1; i < 27; i++) // first bit in mask is 1u<<RIGHT
            {
                if(isSet(i))
                {
                    uint32_t tmp = i;
                    uint32_t tmp_ex = ex;
                    while(tmp_ex >= 3)
                    {
                        tmp_ex /= 3;
                        tmp /= 3;
                    }
                    if(tmp % 3 == tmp_ex)
                        return true;
                }
            }
            return false;
        }

        /**
         * Returns if direction ex is set in this mask.
         *
         * @param ex the direction to query
         * @return true if ex is set, false otherwise
         */
        HDINLINE bool isSet(uint32_t ex) const
        {
            return (bitMask & (1u << ex)) != 0;
        }

        /**
         * Returns a Mask with complementary directions to this one.
         *
         * examples:
         * Mask(LEFT).getMirroredMask() will return Mask(RIGHT)
         * Mask(TOP+LEFT).getMirroredMask() will return Mask(BOTTOM+RIGHT)
         * (Mask(LEFT)+Mask(BACK)).getMirroredMask() will return (Mask(RIGHT)+Mask(FRONT))
         * ...
         *
         * @return newly created mask
         */
        Mask getMirroredMask() const
        {
            uint32_t tmp = 0;
            for(uint32_t i = 1; i < 27; i++) // first bit in mask is 1u<<RIGHT
            {
                if(isSet((ExchangeType) i))
                {
                    tmp |= (1u << getMirroredExchangeType((ExchangeType) i));
                }
            }
            Mask result;
            result.bitMask = tmp;
            return result;
        }

        /**
         * Returns an ExchangeType with complementary directions to this Mask.
         *
         * See Mask::getMirroredMask for examples.
         *
         * @return complementary ExchangeType
         */
        static ExchangeType getMirroredExchangeType(uint32_t ex)
        {
            if(ex >= traits::NumberOfExchanges<DIM3>::value)
                throw std::runtime_error("parameter exceeds allowed maximum");

            Mask mask(ex);
            uint32_t tmp = 0;
            if(mask.containsExchangeType(RIGHT))
                tmp += LEFT;
            if(mask.containsExchangeType(LEFT))
                tmp += RIGHT;
            if(mask.containsExchangeType(BOTTOM))
                tmp += TOP;
            if(mask.containsExchangeType(TOP))
                tmp += BOTTOM;
            if(mask.containsExchangeType(FRONT))
                tmp += BACK;
            if(mask.containsExchangeType(BACK))
                tmp += FRONT;

            return (ExchangeType) tmp;
        }

        /** translate direction to relative offset
         *
         * direction (combination of `ExchangeType`'s) e.g. TOP, TOP+LEFT, ... @see types.h
         * are translated to a relative offsets were every dimension is set to one of -1,0,1
         *                      X      Y         Z
         *  - `-1` if contains LEFT,  TOP    or FRONT
         *  - `+1` if contains RIGHT, BOTTOM or BACK
         *  - `0`  else
         *
         * @param direction combination which describe a direction (only one direction)
         * @return DataSpace with relative offsets
         */
        template<unsigned DIM>
        static HDINLINE DataSpace<DIM> getRelativeDirections(uint32_t direction)
        {
            DataSpace<DIM> tmp;

            for(uint32_t d = 0; d < DIM; ++d)
            {
                const int dim_direction(direction % 3);
                tmp[d] = (dim_direction == 2 ? -1 : dim_direction);
                direction /= 3;
            }
            return tmp;
        }

    protected:
        /**
         * mask which is a combination of the type \see ExchangeType
         */
        uint32_t bitMask;
    };

    /** special implementation for `DIM1`
     *
     * optimization: no modulo is used
     */
    template<>
    HDINLINE DataSpace<DIM1> Mask::getRelativeDirections(uint32_t direction)
    {
        return (direction == 2 ? DataSpace<DIM1>(-1) : DataSpace<DIM1>(direction));
    }

} // namespace pmacc
