/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    template<typename T_Sequence, template<typename> typename T_Accessor = mp_identity_t>
    struct InheritLinearly;

    template<typename... Ts, template<typename> typename T_Accessor>
    struct InheritLinearly<mp_list<Ts...>, T_Accessor> : T_Accessor<Ts>...
    {
    };
} // namespace pmacc
