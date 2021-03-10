// perf.cc
#include "pmacc/eventSystem/Perf.hpp"
#include <vector>
#include <tuple>
#include <map>
#include <cmath>

namespace Performance {
    using d2 = std::tuple<double,double>;

    static bool tracing = false;
    static std::map<std::string,std::map<d2,PerfAvg>> events;

    void Timers::append(const std::string &label, PerfData &datum) {
        if(!tracing) return;
        auto it = events.find(label);
        d2 key(datum.bytes, datum.flops);
        if (it == events.end()) {
            auto v = std::map<d2,PerfAvg>();
            v.emplace(key, PerfAvg(datum.t1-datum.t0));
            events[label] = v;
        } else {
            auto &et = it->second;
            auto ev = et.find(key);
            if (ev == et.end()) {
                et.emplace(key, PerfAvg(datum.t1-datum.t0));
            } else {
                ev->second.add(datum.t1-datum.t0);
            }
        }
    }
    void Timers::on() { tracing = true; }
    void Timers::off() { tracing = false; }

    void Timers::show(std::ostream &os) {
        const char hdr1[] = "{ ";
        const char hdr2[] = ", ";

        const char *bhdr = hdr1;
        for(auto et : events) { // all events for thread
            os << bhdr << "'" << et.first << "' : [" << std::endl;
            bhdr = hdr2;
            for(auto ev : et.second) { // all map values
                auto x = ev.second;
                os << "      { 'Bytes': " << std::get<0>(ev.first) << std::endl
                   << "      , 'Flops': " << std::get<1>(ev.first) << std::endl
                   << "      , 'Calls': " << x.n << std::endl
                   << "      , 'Min': "   << x.min << std::endl
                   << "      , 'Max': "   << x.max << std::endl
                   << "      , 'Avg': "   << x.s/x.n << std::endl
                   << "      , 'Stddev': " << std::sqrt(x.v/x.n) << " }," << std::endl;
            }
            os << "    ]" << std::endl;
        }
        os << "  }" << std::endl;
    }
}
