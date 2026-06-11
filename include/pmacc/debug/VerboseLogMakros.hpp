/*
 * SPDX-FileCopyrightText: Rene Widera, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/debug/VerboseLog.hpp"

#include <string>

/** create a log lvl
 * @param code integer which represent a bit in a 64bit bitmask
 * @param name name of the log lvl, name is needet later to call log<name>(...)
 */
#define DEFINE_LOGLVL(code, name)                                                                                     \
    typedef pmacc::LogLvl<code, thisClass> name;                                                                      \
    friend inline std::string getLogName(const name)                                                                  \
    {                                                                                                                 \
        return std::string(#name);                                                                                    \
    }

/** set a default value for a verbose class
 * @param default_lvl must be a integer which represent a defined log lvl
 */
#define __DEFINE_VERBOSE_CLASS_DEFAULT_LVL(default_lvl)                                                               \
    static constexpr uint64_t log_level = default_lvl;                                                                \
    }

/** helper for define log lvl inside of DEFINE_VERBOSE_CLASS
 */
#define __DEFINE_VERBOSE_CLASS_LVLS(...)                                                                              \
    __VA_ARGS__                                                                                                       \
    __DEFINE_VERBOSE_CLASS_DEFAULT_LVL

/** create a struct which represent a verbose container
 * @param structName name of the container(struct)
 */
#define DEFINE_VERBOSE_CLASS(structName)                                                                              \
    struct structName                                                                                                 \
    {                                                                                                                 \
        static std::string getName()                                                                                  \
        {                                                                                                             \
            return std::string(#structName);                                                                          \
        }                                                                                                             \
                                                                                                                      \
    private:                                                                                                          \
        typedef structName thisClass;                                                                                 \
                                                                                                                      \
    public:                                                                                                           \
        __DEFINE_VERBOSE_CLASS_LVLS
