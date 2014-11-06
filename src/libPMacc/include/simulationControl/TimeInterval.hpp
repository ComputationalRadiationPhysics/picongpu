/**
 * Copyright 2013 Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
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

#ifndef TIMEINTERVAL_HPP
#define	TIMEINTERVAL_HPP

#include <sys/time.h>
#include <string>

#include "types.h"
#include <iostream>
#include <sstream>


namespace PMacc
{

    class TimeIntervall
    {
    public:

        TimeIntervall()
        {
            start=end=getTime();
        }

        /*! Get the timestamp in msec
         * @return time of the moment
         */
        static double getTime()
        {
            struct timeval act_time;
            gettimeofday(&act_time, NULL);
            return (double)act_time.tv_sec*1000. + (double)act_time.tv_usec / 1000.;
        }

        double toggleStart()
        {
            return start=getTime();
        }

        double toggleEnd()
        {
            return end=getTime();
        }

        double getInterval()
        {
            return end-start;
        }

        std::string printInterval()
        {
            return printeTime(getInterval());
        }

        static std::string printeTime(double time)
        {
            std::ostringstream outstr;


            int p_time;

            bool write_all=false;
            if(time/(3600.*1000.)>1.)
            {
                p_time=time/(3600.*1000.);
                time=time-3600.*1000.*p_time;
                outstr<<std::setw(2)<<p_time<<"h ";
                write_all=true;
            }


            if(write_all || time/(60*1000)>1.)
            {
                p_time=time/(60.*1000.);
                time=time-60.*1000.*p_time;
                outstr<<std::setw(2)<<p_time<<"min ";
                write_all=true;
            }


            if(write_all || time/1000.>1.)
            {
                p_time=time/1000.;
                time=time-1000.*p_time;
                outstr<<std::setw(2)<<p_time<<"sec ";
                write_all=true;
            }


            if(write_all || time>1.)
            {
                outstr<<std::setw(3)<<(int)time<<"msec";
            }

            if(outstr.str().empty())
                outstr<<"  0msec";

            return outstr.str();
        }

    private:
        double start;
        double end;
    };
} //namespace PMacc


#endif	/* TIMEINTERVAL_HPP */

