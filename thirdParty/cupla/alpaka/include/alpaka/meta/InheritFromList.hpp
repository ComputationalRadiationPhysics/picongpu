/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace alpaka
{
    namespace meta
    {
        template<typename TBaseList>
        class InheritFromList;

        template<template<typename...> class TList, typename... TBases>
        class InheritFromList<TList<TBases...>> : public TBases...
        {
        };
    } // namespace meta
} // namespace alpaka
