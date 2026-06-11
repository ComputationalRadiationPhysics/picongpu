/*
 * SPDX-FileCopyrightText: Marco Garten
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

/** \file ionizers.hpp
 *
 * Includes containing definition of < Ionization Models >
 * which itself each include their own < Ionization Algorithm >
 * that implements what the model actually DOES
 */

#include "picongpu/particles/ionization/byField/ADK/ADK_Impl.hpp"
#include "picongpu/particles/ionization/byField/BSI/BSI_Impl.hpp"
#include "picongpu/particles/ionization/byField/Keldysh/Keldysh_Impl.hpp"
