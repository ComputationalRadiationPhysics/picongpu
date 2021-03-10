// perf.hh
#ifndef PERF_HH
#define PERF_HH
#include <stdint.h>
#include <iostream>

#ifdef _OPENMP
#  include <omp.h>
#else
// mini-openmp compatibility layer
#include <mpi.h>
inline double omp_get_wtime() {
    return MPI_Wtime();
}
#endif

namespace Performance {
struct PerfData {
    double t0, t1;
    double bytes, flops;
    PerfData(double _t0, uint64_t _bytes, uint64_t _flops)
        : t0(_t0), bytes(_bytes), flops(_flops) {}
};

struct PerfAvg {
    /** Accumulates:
     *
     *   min = min_i x_i
     *   max = max_i x_i
     *   s = \sum_i x_i
     *   v = \sum_i (x_i - s/n)^2
     *   n = \sum_i 1
     */
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

struct Timers {
    /**
     *  This class is declared (global) inside the perf.cc and must exist only there.
     */
    static void append(const std::string &label, PerfData &datum);
    static void show(std::ostream &os);
    static void on();
    static void off();
};

class Timed {
    public:
        Timed(const std::string &_label, uint64_t bytes, uint64_t flops)
            : label(_label), datum(omp_get_wtime(),
                                    bytes, flops) {}
        ~Timed() {
            datum.t1 = omp_get_wtime();
            Timers::append(label, datum);
        }
    private:
        const std::string label;
        PerfData datum;
};
}
#endif
