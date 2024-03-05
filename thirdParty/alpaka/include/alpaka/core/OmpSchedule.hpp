/* Copyright 2022 Sergei Bastrakov, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <cstdint>

namespace alpaka::omp
{
    //! Representation of OpenMP schedule information: kind and chunk size. This class can be used regardless of
    //! whether OpenMP is enabled.
    struct Schedule
    {
        //! Schedule kinds corresponding to arguments of OpenMP schedule clause
        //!
        //! Kinds also present in omp_sched_t enum have the same integer values.
        //! It is enum, not enum class, for shorter usage as omp::Schedule::[kind] and to keep interface of 0.6.0.
        enum Kind
        {
            // Corresponds to not setting schedule
            NoSchedule,
            Static = 1u,
            Dynamic = 2u,
            Guided = 3u,
            // Auto supported since OpenMP 3.0
#if defined _OPENMP && _OPENMP >= 200805
            Auto = 4u,
#endif
            Runtime = 5u
        };

        //! Schedule kind.
        Kind kind;

        //! Chunk size. Same as in OpenMP, value 0 corresponds to default chunk size. Using int and not a
        //! fixed-width type to match OpenMP API.
        int chunkSize;

        //! Create a schedule with the given kind and chunk size
        ALPAKA_FN_HOST constexpr Schedule(Kind myKind = NoSchedule, int myChunkSize = 0)
            : kind(myKind)
            , chunkSize(myChunkSize)
        {
        }
    };

    //! Get the OpenMP schedule that is applied when the runtime schedule is used.
    //!
    //! For OpenMP >= 3.0 returns the value of the internal control variable run-sched-var.
    //! Without OpenMP or with OpenMP < 3.0, returns the default schedule.
    //!
    //! \return Schedule object.
    ALPAKA_FN_HOST inline auto getSchedule()
    {
        // Getting a runtime schedule requires OpenMP 3.0 or newer
#if defined _OPENMP && _OPENMP >= 200805
        omp_sched_t ompKind;
        int chunkSize = 0;
        omp_get_schedule(&ompKind, &chunkSize);
        return Schedule{static_cast<Schedule::Kind>(ompKind), chunkSize};
#else
        return Schedule{};
#endif
    }

    //! Set the OpenMP schedule that is applied when the runtime schedule is used for future parallel regions.
    //!
    //! For OpenMP >= 3.0 sets the value of the internal control variable run-sched-var according to the given
    //! schedule. Without OpenMP or with OpenMP < 3.0, does nothing.
    //!
    //! Note that calling from inside a parallel region does not have an immediate effect.
    ALPAKA_FN_HOST inline void setSchedule(Schedule schedule)
    {
        if((schedule.kind != Schedule::NoSchedule) && (schedule.kind != Schedule::Runtime))
        {
#if defined _OPENMP && _OPENMP >= 200805
            omp_set_schedule(static_cast<omp_sched_t>(schedule.kind), schedule.chunkSize);
#endif
        }
    }
} // namespace alpaka::omp
