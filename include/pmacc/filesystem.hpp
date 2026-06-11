/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#if (PMACC_USE_STD_EXPERIMENTAL_FILESYSTEM == 1)
#    include <experimental/filesystem>
namespace stdfs = std::experimental::filesystem;
#else
#    include <filesystem>
namespace stdfs = std::filesystem;
#endif
