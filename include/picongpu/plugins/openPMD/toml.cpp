/* Copyright 2021-2023 Franz Poeschel
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#if ENABLE_OPENPMD == 1

#    include "picongpu/plugins/openPMD/toml.hpp"

#    include "picongpu/plugins/common/MPIHelpers.hpp"
#    include "picongpu/plugins/misc/containsObject.hpp"
#    include "picongpu/plugins/misc/removeSpaces.hpp"
#    include "picongpu/plugins/misc/splitString.hpp"

#    include <pmacc/communication/manager_common.hpp>
#    include <pmacc/filesystem.hpp>

#    include <chrono>
#    include <sstream>
#    include <thread> // std::this_thread::sleep_for
#    include <utility> // std::forward

#    include <mpi.h>
#    include <toml.hpp>

namespace
{
    using PeriodTable_t = std::vector<picongpu::toml::Periodicity>;

    void mergePeriodTable(PeriodTable_t& into, PeriodTable_t&& from)
    {
        for(auto& periodicity : from)
        {
            into.push_back(std::move(periodicity));
        }
    }

    using toml_t = toml::basic_value<toml::discard_comments>;

    /*
     * In our TOML schema, each data sink may specify a TOML table.
     * This function parses one such table.
     */
    PeriodTable_t parseOnePeriodTable(toml::value::table_type const& periodTable)
    {
        PeriodTable_t res;
        for(auto& pair : periodTable)
        {
            /*
             * Each entry in the table may have one of the following forms:
             * <listOfTimeSlices> = [<dataSource>*]
             * <listOfTimeSlices> = <dataSource>
             *
             * Here, <listOfTimeSlices> is a string-formatted time slice in usual PIConGPU syntax, and <dataSource>
             * is a string referring to an accepted data source.
             */
            std::vector<picongpu::toml::TimeSlice> timeSlices = picongpu::toml::parseTimeSlice(pair.first);

            std::vector<std::string> dataSources;
            using maybe_array_t = toml::result<toml::value::array_type const*, toml::type_error>;
            auto dataSourcesInToml = [&pair]() -> maybe_array_t
            {
                try
                {
                    return toml::ok(&pair.second.as_array());
                }
                catch(toml::type_error const& e)
                {
                    return toml::err(e);
                }
            }();
            if(dataSourcesInToml.is_ok())
            {
                // 1. option: dataSources is an array:
                for(auto& value : *dataSourcesInToml.as_ok())
                {
                    auto dataSource
                        = toml::expect<std::string>(value)
                              .or_else(
                                  [](auto const&) -> toml::success<std::string>
                                  {
                                      throw std::runtime_error("[openPMD plugin] Data sources in TOML "
                                                               "file must be a string or a vector of strings.");
                                  })
                              .value;
                    dataSources.push_back(std::move(dataSource));
                }
            }
            else
            {
                // 2. option: dataSources is no array, check if it is a simple string
                auto dataSource
                    = toml::expect<std::string>(pair.second)
                          .or_else(
                              [](auto const&) -> toml::success<std::string>
                              {
                                  throw std::runtime_error("[openPMD plugin] Data sources in TOML "
                                                           "file must be a string or a vector of strings.");
                              })
                          .value;
                dataSources.push_back(std::move(dataSource));
            }

            for(auto& timeSlice : timeSlices)
            {
                res.emplace_back(picongpu::toml::Periodicity{std::move(timeSlice), dataSources});
            }
        }
        return res;
    }

    void parsePluginParameters(
        picongpu::openPMD::PluginParameters& options,
        toml::value tomlConfig,
        std::vector<picongpu::toml::TomlParameter> tomlParameters)
    {
        auto parseOption = [&tomlConfig](std::string& target, std::string const& key)
        {
            if(!tomlConfig.contains(key))
            {
                return; // leave the default option
            }
            try
            {
                target = toml::find<std::string>(tomlConfig, key);
            }
            catch(toml::type_error const& e)
            {
                throw std::runtime_error(
                    "[openPMD plugin] Global key '" + key + "' must point to a value of string type.");
            }
        };

        for(auto const& [param, target_pointer] : tomlParameters)
        {
            parseOption(options.*target_pointer, param);
        }
    }

    PeriodTable_t parseTomlFile(
        picongpu::toml::DataSources& dataSources,
        std::vector<picongpu::toml::TomlParameter> tomlParameters,
        std::string const& content,
        std::string const& file = "unknown file")
    {
        auto data = [&content, &file]()
        {
            std::istringstream istream(content.c_str());
            return toml::parse(istream, file);
        }();

        parsePluginParameters(dataSources.openPMDPluginParameters, data, tomlParameters);

        if(not data.contains("sink"))
        {
            return {};
        }
        auto& sinkTable = [&data]() -> toml::value::table_type const&
        {
            try
            {
                return toml::find(data, "sink").as_table();
            }
            catch(toml::type_error const& e)
            {
                throw std::runtime_error(
                    "[openPMD plugin] Key 'sink' in TOML file must be a table (" + std::string(e.what()) + ")");
            }
        }();

        PeriodTable_t period;
        for(auto const& pair : sinkTable)
        {
            std::string const& sinkName = pair.first;
            (void) sinkName; // we don't need it for now
            toml::value const& perSink = pair.second;
            if(not perSink.contains("period"))
            {
                continue;
            }
            auto& periodTable = [&perSink]() -> toml::value::table_type const&
            {
                try
                {
                    return toml::find(perSink, "period").as_table();
                }
                catch(toml::type_error const& e)
                {
                    throw std::runtime_error(
                        "[openPMD plugin] Key 'period' in TOML file must be a table (" + std::string(e.what()) + ")");
                }
            }();
            mergePeriodTable(period, parseOnePeriodTable(periodTable));
        }
        return period;
    }

    template<typename ChronoDuration>
    PeriodTable_t waitForAndParseTomlFile(
        picongpu::toml::DataSources& dataSources,
        std::vector<picongpu::toml::TomlParameter> tomlParameters,
        std::string const path,
        ChronoDuration const& sleepInterval,
        ChronoDuration const& timeout,
        MPI_Comm comm)
    {
        int rank;
        MPI_CHECK(MPI_Comm_rank(comm, &rank));
        {
            char const* argsv[] = {path.c_str()};
            picongpu::toml::writeLog("openPMD: Reading data requirements from TOML file: '%1%'", 1, argsv);
        }

        // wait for file to appear
        if(rank == 0)
        {
            using namespace std::literals::chrono_literals;

            char const* argsv[] = {path.c_str()};
            ChronoDuration waitedFor = 0s;
            while(!stdfs::exists(path))
            {
                picongpu::toml::writeLog("openPMD: Still waiting for TOML file:\n\t%1%", 1, argsv);
                if(waitedFor > timeout)
                {
                    std::stringstream errorMsg;
                    std::chrono::seconds const asSeconds = timeout;
                    errorMsg << "openPMD: TOML file '" << path << "' was not found within the timeout of "
                             << asSeconds.count() << " seconds.";
                    throw std::runtime_error(errorMsg.str());
                }
                std::this_thread::sleep_for(sleepInterval);
                waitedFor += sleepInterval;
            }
        }

        MPI_CHECK(MPI_Barrier(comm));

        picongpu::toml::writeLog("openPMD: Reading TOML file collectively");
        std::string fileContents = picongpu::collective_file_read(path, comm);

        return parseTomlFile(dataSources, tomlParameters, fileContents, path);
    }
} // namespace


namespace picongpu
{
    namespace toml
    {
        using namespace std::literals::chrono_literals;
        constexpr std::chrono::seconds const WAIT_TIME = 5s;
        constexpr std::chrono::seconds const TIMEOUT = 5min;

        std::string TimeSlice::asString() const
        {
            return std::to_string(start) + ':' + std::to_string(end) + ':' + std::to_string(period);
        }

        DataSources::DataSources(
            std::string const& tomlFile,
            std::vector<picongpu::toml::TomlParameter> tomlParameters,
            std::vector<std::string> const& allowedDataSources,
            MPI_Comm comm,
            openPMD::PluginParameters openPMDPluginParameters_in)
            : openPMDPluginParameters{std::move(openPMDPluginParameters_in)}
        {
            /*
             * Do NOT put the following line as part of the constructor initializers!
             * It takes *this as first parameter, so things must be fully default-constructed before calling it.
             */
            m_periods = waitForAndParseTomlFile(*this, tomlParameters, tomlFile, WAIT_TIME, TIMEOUT, comm);
            for(auto& periodicity : m_periods)
            {
                for(auto const& source : periodicity.sources)
                {
                    if(!plugins::misc::containsObject(allowedDataSources, source))
                    {
                        throw std::runtime_error("[openPMD plugin]: unknown data source '" + source + "'");
                    }
                }
            }
        }

        std::vector<std::string> DataSources::currentDataSources(SimulationStep_t currentStep) const
        {
            std::set<std::string> result;
            for(Periodicity const& period : m_periods)
            {
                auto const& timeslice = period.timeSlice;
                /*
                 * Check conditions:
                 * 1. Current step is later than the timeslice's start
                 * 2. Current step is generated by the timeslice's period
                 *    Needs to be second condition to avoid underflow
                 *    with the current implementation.
                 * 3. Current step is not later than the timeslice's end
                 */
                if(currentStep >= timeslice.start // condition 1
                   && (currentStep - timeslice.start) % timeslice.period == 0 // condition 2
                   && currentStep <= timeslice.end) // condition 3
                {
                    for(std::string const& source : period.sources)
                    {
                        result.insert(source);
                    }
                }
            }
            return {result.begin(), result.end()};
        }

        std::string DataSources::periods() const
        {
            std::vector<std::string> notConcatenated;
            for(auto const& periodicity : m_periods)
            {
                notConcatenated.push_back(periodicity.timeSlice.asString());
            }
            if(notConcatenated.empty())
            {
                return {};
            }
            std::stringstream stringBuilder;
            auto iterator = notConcatenated.begin();
            stringBuilder << *iterator;
            for(; iterator != notConcatenated.end(); ++iterator)
            {
                stringBuilder << "," << *iterator;
            }
            return stringBuilder.str();
        }
    } // namespace toml
} // namespace picongpu

#endif // ENABLE_OPENPMD
