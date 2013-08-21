/**
 * Copyright 2013 René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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

#include <sstream>
#include <boost/format.hpp>
#include <iostream>

#include "static_assert.hpp"
#include <string>

#include "debug/VerboseLogMakros.hpp"


namespace PMacc
{


/** get the name of a verbose lvl
 * 
 * this function is defined as friend function for every log lvl
 * @param dummy instance of LogClass to find name
 * @return name of LogClass
 */
template<class LogClass>
std::string getLogName(const LogClass& dummy);


namespace verboseLog_detail
{

template<typename X, typename Y>
struct IsSameClassType
{
    static const bool result = false;
};

template<typename X>
struct IsSameClassType<X, X>
{
    static const bool result = true;
};

} //namespace verboseLog_detail

template<uint64_t lvl_, class membership_>
struct LogLvl
{
    typedef membership_ Parent;
    static const uint64_t lvl = lvl_;

    /*this operation is only allowed for LogLvl from the same parent
     * create a LogLvl which has to lvl, only on lvl must be set to be true
     */
    template<class OtherLogLvl >
    LogLvl < (OtherLogLvl::lvl | lvl), membership_> operator+(const OtherLogLvl & other)
    {
        return LogLvl < (OtherLogLvl::lvl | lvl), membership_ > ();
    }

};

namespace verboseLog_detail
{

template<class LogLevel>
class VerboseLog
{
private:
    typedef typename LogLevel::Parent LogParent;
    static const uint64_t logLvl = LogLevel::lvl;
public:

    VerboseLog(const char* msg) : fmt(msg)
    {
    }

    ~VerboseLog()
    {
        typedef LogLvl<(logLvl & LogParent::log_level), LogParent> LogClass;
        /* check if bit in mask is set
         * If you get an linker error in the next two lines you have not used 
         * DEFINE_LOGLVL makro to define a named logLvl
         */
        if (logLvl & LogParent::log_level) /*compiletime check*/
        {
            std::cout << LogParent::getName() << " " << getLogName(LogClass()) <<
            "(" << (logLvl & LogParent::log_level) << ")" << " | " << fmt << std::endl;
        }
    }

    template <typename T>
    VerboseLog& operator %(T value)
    {
        if (logLvl & LogParent::log_level) /*compiletime check*/
            fmt % value;
        return *this;
    }

protected:
    boost::format fmt;
};

}//namespace verboseLog_detail

/*
 * example call:
 * log<MYLOGLEVELS::CRITICAL>("printf %2% stream %1%, number example %3%.") % "messages" % "style" % 5;
 * output of example: 4 | printf style stream messages, number example 5
 */
template <class LogLvl>
static verboseLog_detail::VerboseLog<LogLvl> log(const char* msg)
{
    return verboseLog_detail::VerboseLog<LogLvl > (msg);
}

/* version which allow combine of error levels
 * example call:
 * log(MYLOGLEVELS::CRITICAL+MYLOGLEVELS::MEMORY,"printf %2% stream %1%, number example %3%.") % "messages" % "style" % 5
 */
template <class LogLvl>
static verboseLog_detail::VerboseLog<LogLvl> log(const LogLvl lvl, const char* msg)
{
    return verboseLog_detail::VerboseLog<LogLvl > (msg);
}



} //namespace PMacc


