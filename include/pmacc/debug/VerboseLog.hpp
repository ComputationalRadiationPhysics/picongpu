/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/* Copyright 2013-2024 Rene Widera, Benjamin Worpitz, Alexander Grund
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pmacc/boost_workaround.hpp>

#include <boost/format.hpp>

#include <cstdint>
#include <iostream>
#include <string>

namespace pmacc
{
    /** get the name of a verbose lvl
     *
     * this function is defined as friend function for every log lvl
     * @param dummy instance of LogClass to find name
     * @return name of LogClass
     */
    template<class LogClass>
    std::string getLogName(LogClass const& dummy)
    {
        return std::string("UNDEFINED_LVL");
    }

    template<uint64_t lvl_, class membership_>
    struct LogLvl
    {
        using Parent = membership_;
        static constexpr uint64_t lvl = lvl_;

        /* This operation is only allowed for LogLvl with the same Parent type.
         * Create a LogLvl that contains two levels. At least one lvl has to be true
         */
        template<class OtherLogLvl>
        LogLvl<(OtherLogLvl::lvl | lvl), membership_> operator+(OtherLogLvl const&)
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
            using LogParent = typename LogLevel::Parent;
            static constexpr uint64_t logLvl = LogLevel::lvl;

        public:
            VerboseLog(char const* msg) : fmt(msg)
            {
            }

            ~VerboseLog()
            {
                using LogClass = LogLvl<(logLvl & LogParent::log_level), LogParent>;
                /* check if a bit in the mask is set
                 * If you get an linker error in the next two lines you have not used
                 * DEFINE_LOGLVL makro to define a named logLvl
                 */
                if constexpr(static_cast<bool>(logLvl & LogParent::log_level))
                {
                    std::cout << LogParent::getName() << " " << getLogName(LogClass()) << "("
                              << (logLvl & LogParent::log_level) << ")"
                              << " | " << fmt << std::endl;
                }
            }

            template<typename T>
            VerboseLog& operator%(T value)
            {
                if constexpr(static_cast<bool>(logLvl & LogParent::log_level))
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
    verboseLog_detail::VerboseLog<LogLvl> log(char const* msg)
    {
        return verboseLog_detail::VerboseLog<LogLvl>(msg);
    }

    /* version that allows to combine error levels
     * example call:
     * log(MYLOGLEVELS::CRITICAL+MYLOGLEVELS::MEMORY,"printf %2% stream %1%, number example %3%.") % "messages" %
     * "style" % 5
     */
    template<class LogLvl>
    verboseLog_detail::VerboseLog<LogLvl> log(LogLvl const, char const* msg)
    {
        return verboseLog_detail::VerboseLog<LogLvl>(msg);
    }


} // namespace pmacc
