/* Copyright 2021 David M. Rogers
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

#include <stdint.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <map>
#include <cmath>
#include <chrono>
#include <string>
#include "pmacc/Environment.def"

namespace pmacc
{
    /* Internal data structures for counters, etc. */
    namespace {
        using secs = std::chrono::duration<double, std::ratio<1, 1>>;
        using tpoint_t = std::chrono::time_point<std::chrono::steady_clock>;
        inline tpoint_t walltime() {
            return std::chrono::steady_clock::now();
        }
        // returns value in seconds
        inline double timeDelta(tpoint_t t0, tpoint_t t1) {
            return secs(t1 - t0).count();
        }

        /** Accumulator for:
         *
         *   min = min_i x_i
         *   max = max_i x_i
         *   s = \sum_i x_i
         *   v = \sum_i (x_i - s/n)^2
         *   n = \sum_i 1
         */
        struct PerfAvg {
            PerfAvg(double x) : min(x), max(x), s(x), v(0.0), n(1) { }

            void add(const double x) {
                // Old Trick From: https://manual.gromacs.org/2020/reference-manual/averages.html
                if(x < min) min = x;
                if(x > max) max = x;
                v += (s - n*x)*(s - n*x) / (n*(n+1.0));
                s += x;
                n += 1;
            }
            double min, max, s, v;
            int n;
        };
    }

    /** Simple container for bytes, flops, and times associated with a code region.
     */
    struct PerfData {
        tpoint_t t0, t1;
        double bytes, flops;
        PerfData(double _bytes, double _flops)
            : t0(walltime()), t1(t0), bytes(_bytes), flops(_flops) {}
        void stop() { if(t1 == t0) t1 = walltime(); }; ///< stop-if-not-already-stopped
        bool stopped() const { return t1 != t0; } ///< return true if it has been stopped
        double duration() const { return timeDelta(t0, t1); } ///< return seconds elapsed
    };

    /** Singleton class collecting all PerfData reports.
     *  It's disabled by default.
     *
     *  To use: 
     *    1. run (at code start) PerfInfo::getInstance().on();
     *    2. construct PerfInfo section("section name", bytes, flops);
     *       within blocks you wish to time.
     *    3. optionally (when done) run PerfInfo::getInstance().off();
     *    4. print accumulated information using PerfInfo::getInstance().show(std::cout);
     *
     *  Statistics (#calls, avg. time per call, etc.) are reported separately
     *  for each ("section name", bytes, flops) combination.
     *
     */
    class PerfInfo
    {
    public:
        /** Performance Timers will save into collection when completed. (append is enabled)
         */
        void on()  { tracing = true; }

        /** Performance Timers will not save into collection when completed. (append is disabled)
         */
        void off() { tracing = false; }

        /** Append the datum to the collection (under "section name" = label).
         */
        void append(const std::string &label, const PerfData &datum) {
            if(!tracing || !datum.stopped()) return;
            auto it = events.find(label);
            d2 key(datum.bytes, datum.flops);
            if (it == events.end()) {
                auto v = std::map<d2,PerfAvg>();
                v.emplace(key, PerfAvg(datum.duration()));
                events[label] = v;
            } else {
                auto &et = it->second;
                auto ev = et.find(key);
                if (ev == et.end()) {
                    et.emplace(key, PerfAvg(datum.duration()));
                } else {
                    ev->second.add(datum.duration());
                }
            }
        }

        /** Print out a json-formatted summary of the PerfInfo data collected.
         */
        void show(std::ostream &os) {
            const char hdr1[] = "{ ";
            const char hdr2[] = ", ";

            const char *bhdr = hdr1;
            for(auto et : events) { // all events for thread
                os << bhdr << "\"" << et.first << "\" : [" << std::endl;
                bhdr = hdr2;
                int i = 0;
                for(auto ev : et.second) { // all map values
                    if(i != 0)
                        std::cout << ",\n";
                    i = 1;
                    auto x = ev.second;
                    os << "      { \"Bytes\": " << std::get<0>(ev.first) << std::endl
                       << "      , \"Flops\": " << std::get<1>(ev.first) << std::endl
                       << "      , \"Calls\": " << x.n << std::endl
                       << "      , \"Min\": "   << x.min << std::endl
                       << "      , \"Max\": "   << x.max << std::endl
                       << "      , \"Avg\": "   << x.s/x.n << std::endl
                       << "      , \"Stddev\": " << std::sqrt(x.v/x.n) << " }";
                }
                os << "\n   ]\n";
            }
            os << "}\n";
        }

    private:
        friend struct detail::Environment;
        friend class PerfTimed;

        bool tracing = false;
        using d2 = std::tuple<double,double>;
        /** Data is gathered into name and then grouped by call size (bytes,flops)
         *
         * We could collapse these two into <name,bytes,flops> and then column-ize
         * PerfAvg to make more concise output too...
         *
         * name : { <bytes,flops> : PerfAvg }
         */
        std::map<std::string,std::map<d2,PerfAvg>> events;

        PerfInfo() {}
        PerfInfo(const PerfInfo &) = delete; // copy ctor
        PerfInfo(PerfInfo&&) = delete; // move ctor
        PerfInfo& operator=(const PerfInfo&) = delete; // assign ctor
        PerfInfo& operator=(PerfInfo&) = delete;
        static PerfInfo& getInstance()
        {
            static PerfInfo instance;
            return instance;
        }
    };

    /** Scoped class holding a name and a PerfData timer.
     *  It's the user's responsibility to provide meaningful bytes and flops.
     *  Start/stop timers are set by this class's ctor and dtor.
     */
    class PerfTimed {
        public:
            PerfTimed(const std::string &_label, double bytes, double flops)
                : label(_label), datum(bytes, flops) {}
            ~PerfTimed() {
                datum.stop();
                PerfInfo::getInstance().append(label, datum);
            }
            void stop() { datum.stop(); }; ///< Stop the underlying clock.
        private:
            const std::string label;
            PerfData datum;
    };
}
