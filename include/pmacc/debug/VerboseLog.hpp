/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz, Alexander Grund
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

#include "pmacc/debug/VerboseLogMakros.hpp"

#include <boost/format.hpp>

#include <string>
#include <iostream>
#include <sstream>
#include <cstdint>

namespace pmacc
{
    /** get the name of a verbose lvl
     *
     * this function is defined as friend function for every log lvl
     * @param dummy instance of LogClass to find name
     * @return name of LogClass
     */
    template<class LogClass>
    std::string getLogName(const LogClass& dummy)
    {
        return std::string("UNDEFINED_LVL");
    }


    namespace verboseLog_detail
    {
        template<typename X, typename Y>
        struct IsSameClassType
        {
            static constexpr bool result = false;
        };

        template<typename X>
        struct IsSameClassType<X, X>
        {
            static constexpr bool result = true;
        };

    } // namespace verboseLog_detail

    template<uint64_t lvl_, class membership_>
    struct LogLvl
    {
        typedef membership_ Parent;
        static constexpr uint64_t lvl = lvl_;

        /* This operation is only allowed for LogLvl with the same Parent type.
         * Create a LogLvl that contains two levels. At least one lvl has to be true
         */
        template<class OtherLogLvl>
        LogLvl<(OtherLogLvl::lvl | lvl), membership_> operator+(const OtherLogLvl&)
        {
            return LogLvl<(OtherLogLvl::lvl | lvl), membership_>();
        }
    };

    namespace verboseLog_detail
    {
        template<class LogLevel>
        class VerboseLog
        {
        private:
            typedef typename LogLevel::Parent LogParent;
            static constexpr uint64_t logLvl = LogLevel::lvl;

        public:
            VerboseLog(const char* msg) : fmt(msg)
            {
            }

            ~VerboseLog()
            {
                typedef LogLvl<(logLvl & LogParent::log_level), LogParent> LogClass;
                /* check if a bit in the mask is set
                 * If you get an linker error in the next two lines you have not used
                 * DEFINE_LOGLVL makro to define a named logLvl
                 */
                if(logLvl & LogParent::log_level) /*compile-time check*/
                {
                    std::cout << LogParent::getName() << " " << getLogName(LogClass()) << "("
                              << (logLvl & LogParent::log_level) << ")"
                              << " | " << fmt << std::endl;
                }
            }

            template<typename T>
            VerboseLog& operator%(T value)
            {
                if(logLvl & LogParent::log_level) /*compile-time check*/
                    fmt % value;
                return *this;
            }

        protected:
            boost::format fmt;
        };

    } // namespace verboseLog_detail

    /*
     * example call:
     * log<MYLOGLEVELS::CRITICAL>("printf %2% stream %1%, number example %3%.") % "messages" % "style" % 5;
     * output of example: 4 | printf style stream messages, number example 5
     */
    template<class LogLvl>
    verboseLog_detail::VerboseLog<LogLvl> log(const char* msg)
    {
        return verboseLog_detail::VerboseLog<LogLvl>(msg);
    }

    /* version that allows to combine error levels
     * example call:
     * log(MYLOGLEVELS::CRITICAL+MYLOGLEVELS::MEMORY,"printf %2% stream %1%, number example %3%.") % "messages" %
     * "style" % 5
     */
    template<class LogLvl>
    verboseLog_detail::VerboseLog<LogLvl> log(const LogLvl, const char* msg)
    {
        return verboseLog_detail::VerboseLog<LogLvl>(msg);
    }


} // namespace pmacc
