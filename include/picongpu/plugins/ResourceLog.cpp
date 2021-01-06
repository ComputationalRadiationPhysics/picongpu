/* Copyright 2016-2021 Erik Zenker, Axel Huebl
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

// Boost
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>

// STL
#include <string> /* std::string */
#include <sstream> /* std::stringstream */
#include <map> /* std::map */
#include <stdexcept> /* std::runtime_error */

// C LIB
#include <stdint.h> /* uint32_t */


namespace picongpu
{
    namespace detail
    {
        std::string writeMapToPropertyTree(std::map<std::string, size_t> valueMap, std::string outputFormat)
        {
            // Create property tree which contains the resource information
            using boost::property_tree::ptree;
            ptree pt;

            for(auto it = valueMap.begin(); it != valueMap.end(); ++it)
            {
                pt.put(it->first, it->second);
            }

            // Write property tree to string stream
            std::stringstream ss;
            if(outputFormat == "json")
            {
                write_json(ss, pt, false);
            }
            else if(outputFormat == "jsonpp")
            {
                write_json(ss, pt, true);
            }
            else if(outputFormat == "xml")
            {
                write_xml(ss, pt);
            }
            else if(outputFormat == "xmlpp")
            {
                write_xml(ss, pt, boost::property_tree::xml_writer_make_settings<std::string>('\t', 1));
            }
            else
            {
                throw std::runtime_error(
                    std::string("resourcelog.format ") + outputFormat
                    + std::string(" is not known, use json or xml."));
            }

            return ss.str();
        }
    } // namespace detail
} // namespace picongpu
