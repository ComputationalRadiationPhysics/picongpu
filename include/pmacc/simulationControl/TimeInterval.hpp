/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz
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

#include "pmacc/types.hpp"

#include <chrono>
#include <string>
#include <iostream>
#include <sstream>


namespace pmacc
{
    class TimeIntervall
    {
    public:
        TimeIntervall()
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
            return printeTime(getInterval());
        }

        static std::string printeTime(double time)
        {
            std::ostringstream outstr;


            int p_time;

            bool write_all = false;
            if(time / (3600. * 1000.) > 1.)
            {
                p_time = time / (3600. * 1000.);
                time = time - 3600. * 1000. * p_time;
                outstr << std::setw(2) << p_time << "h ";
                write_all = true;
            }


            if(write_all || time / (60 * 1000) > 1.)
            {
                p_time = time / (60. * 1000.);
                time = time - 60. * 1000. * p_time;
                outstr << std::setw(2) << p_time << "min ";
                write_all = true;
            }


            if(write_all || time / 1000. > 1.)
            {
                p_time = time / 1000.;
                time = time - 1000. * p_time;
                outstr << std::setw(2) << p_time << "sec ";
                write_all = true;
            }


            if(write_all || time > 1.)
            {
                outstr << std::setw(3) << (int) time << "msec";
            }

            if(outstr.str().empty())
                outstr << "  0msec";

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
