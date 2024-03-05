/* Copyright 2022 Benjamin Worpitz, Daniel Vollmer, Erik Zenker, René Widera, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"

#if BOOST_OS_WINDOWS || BOOST_OS_CYGWIN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
// We could use some more macros to reduce the number of sub-headers included, but this would restrict user code.
#    include <windows.h>
#elif BOOST_OS_UNIX || BOOST_OS_MACOS
#    include <sys/param.h>
#    include <sys/types.h>
#    include <unistd.h>

#    include <cstdint>
#    if BOOST_OS_BSD || BOOST_OS_MACOS
#        include <sys/sysctl.h>
#    endif
#endif

#if BOOST_OS_LINUX
#    include <fstream>
#endif

#include <cstring>
#include <stdexcept>
#include <string>

#if BOOST_ARCH_X86
#    if BOOST_COMP_GNUC || BOOST_COMP_CLANG || BOOST_COMP_PGI
#        include <cpuid.h>
#    elif BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#        include <intrin.h>
#    endif
#endif

namespace alpaka::cpu::detail
{
    constexpr int NO_CPUID = 0;
    constexpr int UNKNOWN_CPU = 0;
    constexpr int UNKNOWN_COMPILER = 1;
#if BOOST_ARCH_X86
#    if BOOST_COMP_GNUC || BOOST_COMP_CLANG || BOOST_COMP_PGI
    inline auto cpuid(std::uint32_t level, std::uint32_t subfunction, std::uint32_t ex[4]) -> void
    {
        __cpuid_count(level, subfunction, ex[0], ex[1], ex[2], ex[3]);
    }

#    elif BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
    inline auto cpuid(std::uint32_t level, std::uint32_t subfunction, std::uint32_t ex[4]) -> void
    {
        __cpuidex(reinterpret_cast<int*>(ex), level, subfunction);
    }
#    else
    inline auto cpuid(std::uint32_t, std::uint32_t, std::uint32_t ex[4]) -> void
    {
        ex[0] = ex[2] = ex[3] = NO_CPUID;
        ex[1] = UNKNOWN_COMPILER;
    }
#    endif
#else
    inline auto cpuid(std::uint32_t, std::uint32_t, std::uint32_t ex[4]) -> void
    {
        ex[0] = ex[2] = ex[3] = NO_CPUID;
        ex[1] = UNKNOWN_CPU;
    }
#endif
    //! \return The name of the CPU the code is running on.
    inline auto getCpuName() -> std::string
    {
        // Get extended ids.
        std::uint32_t ex[4] = {0};
        cpuid(0x8000'0000, 0, ex);
        std::uint32_t const nExIds(ex[0]);

        if(!nExIds)
        {
            switch(ex[1])
            {
            case UNKNOWN_COMPILER:
                return "<unknown: compiler>";
            case UNKNOWN_CPU:
                return "<unknown: CPU>";
            default:
                return "<unknown>";
            }
        }
#if BOOST_ARCH_X86
        // Get the information associated with each extended ID.
        char cpuBrandString[0x40] = {0};
        for(std::uint32_t i(0x8000'0000); i <= nExIds; ++i)
        {
            cpuid(i, 0, ex);

            // Interpret CPU brand string and cache information.
            if(i == 0x8000'0002)
            {
                std::memcpy(cpuBrandString, ex, sizeof(ex));
            }
            else if(i == 0x8000'0003)
            {
                std::memcpy(cpuBrandString + 16, ex, sizeof(ex));
            }
            else if(i == 0x8000'0004)
            {
                std::memcpy(cpuBrandString + 32, ex, sizeof(ex));
            }
        }
        return std::string(cpuBrandString);
#else
        return std::string("unknown");
#endif
    }

    //! \return Pagesize in bytes used by the system.
    inline size_t getPageSize()
    {
#if BOOST_OS_WINDOWS || BOOST_OS_CYGWIN
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return si.dwPageSize;
#elif BOOST_OS_UNIX || BOOST_OS_MACOS
#    if defined(_SC_PAGESIZE)
        return static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#    else
        // this is legacy and only used as fallback
        return = static_cast<size_t>(getpagesize());
#    endif
#else
#    error "getPageSize not implemented for this system!"
        return 0;
#endif
    }

    //! \return The total number of bytes of global memory.
    //! Adapted from David Robert Nadeau:
    //! http://nadeausoftware.com/articles/2012/09/c_c_tip_how_get_physical_memory_size_system
    inline auto getTotalGlobalMemSizeBytes() -> std::size_t
    {
#if BOOST_OS_WINDOWS
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        return static_cast<std::size_t>(status.ullTotalPhys);

#elif BOOST_OS_CYGWIN
        // New 64-bit MEMORYSTATUSEX isn't available.
        MEMORYSTATUS status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatus(&status);
        return static_cast<std::size_t>(status.dwTotalPhys);

#elif BOOST_OS_UNIX || BOOST_OS_MACOS
        // Unix : Prefer sysctl() over sysconf() except sysctl() with HW_REALMEM and HW_PHYSMEM which are not
        // always reliable
#    if defined(CTL_HW) && (defined(HW_MEMSIZE) || defined(HW_PHYSMEM64))
        int mib[2]
            = { CTL_HW,
#        if defined(HW_MEMSIZE) // OSX
                HW_MEMSIZE
#        elif defined(HW_PHYSMEM64) // NetBSD, OpenBSD.
                HW_PHYSMEM64
#        endif
              };
        std::uint64_t size(0);
        std::size_t sizeLen{sizeof(size)};
        if(sysctl(mib, 2, &size, &sizeLen, nullptr, 0) < 0)
            throw std::logic_error("getTotalGlobalMemSizeBytes failed calling sysctl!");
        return static_cast<std::size_t>(size);

#    elif defined(_SC_AIX_REALMEM) // AIX.
        return static_cast<std::size_t>(sysconf(_SC_AIX_REALMEM)) * static_cast<std::size_t>(1024);

#    elif defined(_SC_PHYS_PAGES) // Linux, FreeBSD, OpenBSD, Solaris.
        return static_cast<std::size_t>(sysconf(_SC_PHYS_PAGES)) * getPageSize();

#    elif defined(CTL_HW)                                                                                             \
        && (defined(HW_PHYSMEM) || defined(HW_REALMEM)) // FreeBSD, DragonFly BSD, NetBSD, OpenBSD, and OSX.
        int mib[2]
            = { CTL_HW,
#        if defined(HW_REALMEM) // FreeBSD.
                HW_REALMEM
#        elif defined(HW_PYSMEM) // Others.
                HW_PHYSMEM
#        endif
              };
        std::uint32_t size(0);
        std::size_t const sizeLen{sizeof(size)};
        if(sysctl(mib, 2, &size, &sizeLen, nullptr, 0) < 0)
            throw std::logic_error("getTotalGlobalMemSizeBytes failed calling sysctl!");
        return static_cast<std::size_t>(size);
#    endif

#else
#    error "getTotalGlobalMemSizeBytes not implemented for this system!"
#endif
    }

    //! \return The free number of bytes of global memory.
    //! \throws std::logic_error if not implemented on the system and std::runtime_error on other errors.
    inline auto getFreeGlobalMemSizeBytes() -> std::size_t
    {
#if BOOST_OS_WINDOWS
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        return static_cast<std::size_t>(status.ullAvailPhys);
#elif BOOST_OS_LINUX
#    if defined(_SC_AVPHYS_PAGES)
        return static_cast<std::size_t>(sysconf(_SC_AVPHYS_PAGES)) * getPageSize();
#    else
        // this is legacy and only used as fallback
        return static_cast<std::size_t>(get_avphys_pages()) * getPageSize();
#    endif
#elif BOOST_OS_MACOS
        int free_pages = 0;
        std::size_t len = sizeof(free_pages);
        if(sysctlbyname("vm.page_free_count", &free_pages, &len, nullptr, 0) < 0)
        {
            throw std::logic_error("getFreeGlobalMemSizeBytes failed calling sysctl(vm.page_free_count)!");
        }

        return static_cast<std::size_t>(free_pages) * getPageSize();
#else
#    error "getFreeGlobalMemSizeBytes not implemented for this system!"
#endif
    }

} // namespace alpaka::cpu::detail
