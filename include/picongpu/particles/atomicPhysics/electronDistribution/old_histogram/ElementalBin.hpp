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
 * This file describes a single bin containing a value for an interval ending with
 * endBoundary(defined in Bin.hpp), as a subclass of the class Bin.
 *
 * The reference implementation cast the whole T_Object object to T_DataType upon
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
 *   - value    ... actual storage of value
 *
 *  public.
 *   - ElementalBin(T_Argument endBoundary) ... constructor
 *   - getValue():T_DataType                ... return value of this Bin
 *   - getEndBoundary():T_Argument  ... return the end Boundary of this Bin
 *   - isEmpty():void               ... returns true if bin has value of 0
 *   - binObject(T_Object& object):void ... adds value of object to histogram bin
 *   - isMultiple():bool            ... return false since an ElementalBin never
 *                                      contains more than 1 value
 */

#pragma once


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
                    class ElementalBin : public Bin<T_DataType, T_Object, T_Argument>
                    {
                    private:
                        T_DataType value; // actual storage of value of histogram

                    public:
                        ElementalBin(T_Argument endBoundary) : Bin<T_DataType, T_Object, T_Argument>(endBoundary)
                        {
                            /** Constructor of ElementalBin, initialises value with zero and
                             * endBoundary by calling the super class
                             */

                            this->value = static_cast<T_DataType>(0);
                        }

                        T_DataType getValue()
                        {
                            /** returns the value of this bin */
                            return this->value;
                        }

                        void binObject(const T_Object& object, T_Argument argument)
                        {
                            /** This is used to add objects value to the bin, standard implementation
                             * simply casts object to the data type of value, specialisations should
                             * cover there specific use cases
                             */
                            this->value += static_cast<T_DataValue>(object);
                        }

                        bool isEmpty()
                        {
                            /** this function returns true if value is 0
                             */
                            if(this->value == static_cast<T_DataType>(0))
                            {
                                return true;
                            }

                            return false
                        }

                        bool isMultiple()
                        {
                            return true;
                        }
                    }

                } // namespace histogram
            } // namespace electronDistribution
        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
