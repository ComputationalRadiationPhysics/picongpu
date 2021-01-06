/* Copyright 2017-2021 Rene Widera
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

#include <vector>
#include <exception>
#include <string>
#include <sstream>


namespace picongpu
{
    namespace plugins
    {
        namespace multi
        {
            /** multi option storage
             *
             * This option stores the values of a multi command line option
             * and allows to set a default value.
             *
             * @tparam T_ValueType type of the option
             */
            template<typename T_ValueType>
            struct Option : public std::vector<T_ValueType>
            {
                using StorageType = std::vector<T_ValueType>;

                //! type of the value
                using ValueType = T_ValueType;


                /** create a option with a default value
                 *
                 * @param name name of the option
                 * @param description description for the option
                 * @param defaultValue default value of the option
                 */
                Option(std::string const& name, std::string const& description, ValueType const& defaultValue)
                    : m_name(name)
                    , m_description(description)
                    , m_defaultValue(defaultValue)
                    , m_hasDefaultValue(true)
                {
                }

                /** create a option without a default value
                 *
                 * @param name name of the option
                 * @param description description for the option
                 */
                Option(std::string const& name, std::string const& description)
                    : m_name(name)
                    , m_description(description)
                    , m_hasDefaultValue(false)
                {
                }

                /** get the name of the option
                 *
                 * @return name
                 */
                std::string getName()
                {
                    return m_name;
                }

                /** get the description of the option
                 *
                 * @return description
                 */
                std::string getDescription()
                {
                    return m_description;
                }

                /** register the option
                 *
                 * @param desc option object where the option is appended
                 * @param prefix prefix to add to the option name
                 * @param additionalDescription extent the default description
                 */
                void registerHelp(
                    boost::program_options::options_description& desc,
                    std::string const& prefix = std::string{},
                    std::string const& additionalDescription = std::string{})
                {
                    std::string printDefault;
                    if(m_hasDefaultValue)
                        printDefault = std::string(" | default: ") + getDefaultAsStr();

                    desc.add_options()(
                        (prefix + "." + getName()).c_str(),
                        boost::program_options::value(getStorage())->multitoken(),
                        (getDescription() + additionalDescription + printDefault).c_str());
                }

                /** get the default value
                 *
                 * Throw an exception if there is no default value defined.
                 *
                 * @param get the default value defined during the construction of this class
                 */
                T_ValueType getDefault()
                {
                    if(!m_hasDefaultValue)
                        throw std::runtime_error(
                            std::string("There is no default value defined for the option: ") + getName());
                    return m_defaultValue;
                }

                /** set a default value
                 *
                 * The old default value will be overwritten if already exists.
                 *
                 * @param defaultValue new default value
                 */
                void setDefault(T_ValueType const& defaultValue)
                {
                    m_hasDefaultValue = true;
                    m_defaultValue = defaultValue;
                }

                //! get the default value as string
                std::string getDefaultAsStr()
                {
                    std::stringstream ss;
                    ss << getDefault();
                    return ss.str();
                }

                /** get the value set by the user
                 *
                 * Throw an exception if there is no default value defined and idx is
                 * larger than the number of options provided by the user.
                 *
                 * @param idx index of the multi plugin
                 * @return if number of user provided option <= idx then the user defined
                 *         value else the default value if defined
                 */
                T_ValueType get(uint32_t idx)
                {
                    if(StorageType::size() <= idx)
                    {
                        if(!m_hasDefaultValue)
                            throw std::runtime_error(std::string(
                                "There is no default value defined for the option " + getName()
                                + " and idx is out of range"));
                        return m_defaultValue;
                    }

                    return StorageType::operator[](idx);
                }

            private:
                std::string const m_name;
                std::string const m_description;

                T_ValueType m_defaultValue;
                bool m_hasDefaultValue = false;

                StorageType* getStorage()
                {
                    return static_cast<StorageType*>(this);
                }
            };

        } // namespace multi
    } // namespace plugins
} // namespace picongpu
