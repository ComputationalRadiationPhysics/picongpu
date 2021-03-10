/* Copyright 2020 Brian Marre
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


/** @file
 * This file describes a composite bin, a number of adjacent bins with same bin width
 * , a subclass of the calss Bin.
 * ending at endBoundary (defined in Bin.hpp).
 * The reference implementation casts the whole T_Object object to T_DataType upon
 * calling binObject and adds the result to the bin value. Other behavours should be
 * implemented as partial specializations.
 *
 *Template Parameters:
 *  T_DataType  ... data Type of stored value
 *  T_Object,   ... type of objects to be binned
 *  T_Argument  ... data type of argument space
 *
 * Members:
 *  private:
 *   - values    ... actual storage of values in a sinlge linked list
 *
 *  public.
 *   - CompositeBin(T_Argument endBoundary, T_Argument width) ... constructor
 *   - getValue():list<T_DataType>  ... return this->values
 *   - getEndBoundary():T_Argument  ... return the end Boundary of this Bin
 *   - isEmpty():void               ... returns true if bin has value of 0
 *   - binObject(T_Object& object):void ... adds value of object to histogram
 *   - isMultiple():bool            ... returns true since CompositeBins always
 *                                      contains more
 */

#pragma once

#include <forward_list>
#inlcude < array>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            namespace electronDistribution
            {
                namespace histogram
                {
                    template<typename T_DataType, typename T_Object, typename T_Argument>
                    class CompositeBin : public Bin<T_DataType, T_Object, T_Argument>
                    {
                    private:
                        std::array<T_DataType, 2> values = {0}; // initiliased with 0
                        T_Argument width;

                    public:
                        CompositeBin(T_Argument endBoundary, T_Argument width)
                        {
                            /** standard constructor values are initialized with zero */
                            this->width = width;
                        }

                        CompositeBin(T_Argument endBoundary, T_Argument width, T_DataType value1, T_DataType value2)
                        {
                            /** if bin values are known before hand use this constructro instead, to
                             * directly initialize the values directly
                             */
                            this->values[0] = value1;
                            this->values[1] = value2;

                            this->width = width;
                        }

                        T_DataType getValue(std::size_t n = 0)
                        {
                            return values[n];
                        }

                        constexpr bool isMultiple()
                        {
                            return true;
                        }

                        void binObject(const T_Object& object, T_Argument argument)
                        {
                            const T_DataType value = static_cast<T_DataType>(object);
                            std::size_t i;

                            i = static_cast<std::size_t>(this->endBoundary - argument)

                                    this->values[i]
                                += value;
                        }
