/* Copyright 2016-2021 Rene Widera
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

#include <map>
#include <string>
#include <iostream>

namespace pmacc
{
    namespace traits
    {
        /** a property with sub properties
         *
         * This class inherit from `std::map`.
         * If the `operator[]` is used to access a not existing key an empty StringProperty
         * with the given key is inserted (default behavior of `std::map`)
         *
         * Key naming convention:
         *     "name" for name, openPMD-compatible when possible
         *     "param" for additional parameters, corresdponding to openPMD
         *             ...Parameters attribute
         */
        struct StringProperty : public std::map<std::string, StringProperty>
        {
            typedef std::map<std::string, StringProperty> StringPropertyMap;

            //! empty constructor
            StringProperty()
            {
            }

            /** constructor
             *
             * creates a property with one key value
             *
             * \param key name of the key
             * \param propertyValue value of the property
             */
            StringProperty(const std::string& key, const std::string& propertyValue) : value(propertyValue)
            {
                (*this)[key] = propertyValue;
            }

            /** overwrite the value from a property
             *
             * \param propertyValue new value
             * \return the property itself
             */
            StringProperty& operator=(const std::string& propertyValue)
            {
                value = propertyValue;
                return *this;
            }

            //! stores a property value
            std::string value;
        };

        /** stream operator for a StringProperty
         */
        HINLINE std::ostream& operator<<(std::ostream& out, const StringProperty& property)
        {
            out << property.value;
            return out;
        }

        /** Get a property tree of an object
         *
         * specialize this struct including the static method `StringProperty get()`
         * to define a property for an object without the method `getStringProperties()`
         *
         * \tparam T_Type any type
         * \return \p T_Type::getStringProperties() if trait `GetStringProperties<>` is not specialized
         */
        template<typename T_Type>
        struct StringProperties
        {
            static StringProperty get()
            {
                return T_Type::getStringProperties();
            }
        };


        /** get the properties of an object
         *
         * The struct `StringProperties<>` needs to be specialized to change the result
         * of this trait for a user defined type.
         * If there is no user defined specialization available this trait inherits from
         * the result of `::getStringProperties()` from the queried type.
         */
        template<typename T_Type>
        struct GetStringProperties : public StringProperty
        {
            GetStringProperties() : StringProperty(StringProperties<T_Type>::get())
            {
            }
        };

        /** get the properties of an object instance
         *
         * same as `GetStringProperties<>` but accepts an instance instead a type
         *
         * \param an instance that shall be queried
         * \return StringProperty of the given instance
         */
        template<typename T_Type>
        HINLINE StringProperty getStringProperties(const T_Type&)
        {
            return GetStringProperties<T_Type>()();
        }

    } // namespace traits
} // namespace pmacc
