/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::meta
{
    template<typename TBaseList>
    class InheritFromList;

    template<template<typename...> class TList, typename... TBases>
    class InheritFromList<TList<TBases...>> : public TBases...
    {
    };
} // namespace alpaka::meta
