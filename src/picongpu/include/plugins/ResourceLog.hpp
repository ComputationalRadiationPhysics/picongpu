/**
 * Copyright 2016 Erik Zenker
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
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

#pragma once

// PMacc
#include "Environment.hpp"
#include "mappings/simulation/ResourceMonitor.hpp"

// PIConGPU
#include "plugins/ILightweightPlugin.hpp"
#include "ILightweightPlugin.hpp"
#include "simulation_defines.hpp"

// Boost
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/filesystem.hpp>

// STL
#include <iostream>  /* std::cout, std::ostream */
#include <numeric>   /* std::accumulate */
#include <string>    /* std::string */
#include <sstream>   /* std::stringstream */
#include <fstream>   /* std::filebuf */
#include <map>       /* std::map */

// C LIB
#include <stdlib.h> /* itoa */
#include <stdint.h> /* uint32_t */

namespace picongpu
{
    using namespace PMacc;


    class ResourceLog : public ILightweightPlugin
    {
    private:
        MappingDesc *cellDescription;
        ResourceMonitor<simDim> resourceMonitor;

        // programm options
        std::string outputFilePrefix;
        std::string streamType;
        std::string outputFormat;
        std::vector<std::string> properties;

        std::filebuf fileBuf;
        std::map<std::string, bool> propertyMap;

    public:

        ResourceLog() :
                cellDescription(NULL)
        {
            Environment<>::get().PluginConnector().registerPlugin(this);
        }

        std::string pluginGetName() const
        {
            return "ResourceLog";
        }

        void notify(uint32_t currentStep)
        {
            //
            // Create property tree which contains the resource information
            using boost::property_tree::ptree;
            ptree pt;

            if(contains(propertyMap, "rank"))
            {
                size_t rank = static_cast<size_t>(Environment<simDim>::get().GridController().getGlobalRank());
                pt.put("resourceLog.rank", rank);
            }

            if(contains(propertyMap,"position"))
            {
                DataSpace<simDim> currentPosition = Environment<simDim>::get().GridController().getPosition();
                pt.put("resourceLog.position.x", currentPosition[0]);
                pt.put("resourceLog.position.y", currentPosition[1]);
                pt.put("resourceLog.position.z", currentPosition[2]);
            }

            if(contains(propertyMap, "currentStep"))
            {
                pt.put("resourceLog.currentStep", currentStep);
            }

            if(contains(propertyMap, "cellCount"))
            {
                size_t cellCount = resourceMonitor.getCellCount();
                pt.put("resourceLog.cellCount", cellCount);
            }

            if(contains(propertyMap,"particleCount"))
            {
                std::vector<size_t> particleCounts = resourceMonitor.getParticleCounts<VectorAllSpecies>(*cellDescription);
                pt.put("resourceLog.particleCount", std::accumulate(particleCounts.begin(), particleCounts.end(), 0));
            }

            //
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
                throw std::runtime_error(std::string("resourcelog.format ") + outputFormat + std::string(" is not known, use json or xml."));
            }

            //
            // Write property tree to the output stream
            if(streamType == "stdout")
            {
                std::cout << ss.str();
            }
            else if (streamType == "stderr")
            {
                std::cerr << ss.str();
            }
            else if (streamType == "file")
            {
                std::ostream os(&fileBuf);
                os << ss.str();
            }
            else
            {
                throw std::runtime_error(std::string("resourcelog.stream ") + streamType + std::string(" is not known, use stdout, stderr or file instead."));
            }
        }

        void pluginRegisterHelp(po::options_description& desc)
        {
            /* register command line parameters for your plugin */
            desc.add_options()
                    ("resourceLog.period", po::value<uint32_t>(&notifyPeriod)->default_value(0),
                     "Enable ResourceLog plugin [for each n-th step]")
                    ("resourceLog.prefix", po::value<std::string>(&outputFilePrefix)->default_value("resourceLog_"),
                     "Set the filename prefix for output file if a filestream was selected")
                    ("resourceLog.stream", po::value<std::string>(&streamType)->default_value("file"),
                     "Output stream [stdout, stderr, file]")
                    ("resourceLog.properties", po::value<std::vector<std::string> >(&properties)->multitoken(),
                     "List of properties to log [rank, position, currentStep, cellCount, particleCount]")
                    ("resourceLog.format", po::value<std::string>(&outputFormat)->default_value("json"),
                     "Output format of log (pp for pretty print) [json, jsonpp, xml, xmlpp]");
        }

        void setMappingDescription(MappingDesc *cellDescription)
        {
            this->cellDescription = cellDescription;
        }

    private:
        uint32_t notifyPeriod;

        void pluginLoad() {
            if(notifyPeriod != 0) {
                Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);

                // Set default resources to log
                if (properties.empty()) {
                    properties.push_back("rank");
                    properties.push_back("position");
                    properties.push_back("currentStep");
                    properties.push_back("cellCount");
                    properties.push_back("particleCount");
                    propertyMap["rank"] = true;
                    propertyMap["position"] = true;
                    propertyMap["currentStep"] = true;
                    propertyMap["particleCount"] = true;
                    propertyMap["cellCount"] = true;
                }
                else {
                    for (size_t i = 0; i < properties.size(); ++i) {
                        propertyMap[properties[i]] = true;
                    }
                }

                // Prepare file for output stream
                if (streamType == "file") {
                    size_t rank = static_cast<size_t>(Environment<simDim>::get().GridController().getGlobalRank());
                    std::stringstream ss;
                    ss << outputFilePrefix << rank;
                    boost::filesystem::path resourceLogPath(ss.str());
                    fileBuf.open(resourceLogPath.string().c_str(), std::ios::out);
                }
            }
        }

        void pluginUnload()
        {
            if(fileBuf.is_open()){
                fileBuf.close();
            }
            /* called when plugin is unloaded, cleanup here */
        }

        template <typename T_MAP>
        bool contains(T_MAP const map, std::string const value)
        {
            return (map.find(value) != map.end());
        }

    };
}

#include "mappings/simulation/ResourceMonitor.tpp"
