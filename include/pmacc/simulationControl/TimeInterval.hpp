/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace pmacc
{
    class TimeInterval
    {
    public:
        TimeInterval()
        {
            start = end = getTime();
        }

        /*! Get the timestamp in msec
         * @return time of the moment
         */
        static double getTime()
        {
            auto time(Clock::now().time_since_epoch());
            auto timestamp = std::chrono::duration_cast<Milliseconds>(time).count();
            return static_cast<double>(timestamp);
        }

        double toggleStart()
        {
            return start = getTime();
        }

        double toggleEnd()
        {
            return end = getTime();
        }

        double getInterval()
        {
            return end - start;
        }

        std::string printInterval()
        {
            return printTime(getInterval());
        }

        static std::string printTime(double time_ms)
        {
            std::chrono::hh_mm_ss time_split{std::chrono::milliseconds(static_cast<long long>(time_ms))};

            auto const h = time_split.hours().count();
            auto const m = time_split.minutes().count();
            auto const s = time_split.seconds().count();
            auto const ms = time_split.subseconds().count();

            std::ostringstream outstr;

            if(h > 0)
            {
                outstr << std::setw(2) << h << "h " << std::setw(2) << m << "min " << std::setw(2) << s << "sec "
                       << std::setw(3) << ms << "msec";
            }
            else if(m > 0)
            {
                outstr << std::setw(2) << m << "min " << std::setw(2) << s << "sec " << std::setw(3) << ms << "msec";
            }
            else if(s > 0)
            {
                outstr << std::setw(2) << s << "sec " << std::setw(3) << ms << "msec";
            }
            else
            {
                outstr << std::setw(3) << ms << "msec";
            }

            return outstr.str();
        }

    private:
        using Clock = std::chrono::high_resolution_clock;
        template<class Duration>
        using TimePoint = std::chrono::time_point<Clock, Duration>;
        using Milliseconds = std::chrono::milliseconds;

        double start;
        double end;
    };
} // namespace pmacc
