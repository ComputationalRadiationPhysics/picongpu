// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code
//
// Cupla version created by Jeff Young in 2021
// Ported from cupla to alpaka by Bernhard Manfred Gruber in 2022

#pragma once

#include "Stream.h"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <vector>

inline constexpr auto IMPLEMENTATION_STRING = "alpaka";

using Dim = alpaka::DimInt<1>;
using Idx = int;
using Vec = alpaka::Vec<Dim, Idx>;
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

template<typename T>
struct AlpakaStream : Stream<T>
{
    AlpakaStream(Idx arraySize, Idx deviceIndex);

    void copy() override;
    void add() override;
    void mul() override;
    void triad() override;
    void nstream() override;
    auto dot() -> T override;

    void init_arrays(T initA, T initB, T initC) override;
    void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

    using PlatformHost = alpaka::PlatformCpu;
    using DevHost = alpaka::Dev<PlatformHost>;
    using PlatformAcc = alpaka::Platform<Acc>;
    using DevAcc = alpaka::Dev<Acc>;
    using BufHost = alpaka::Buf<alpaka::DevCpu, T, Dim, Idx>;
    using BufAcc = alpaka::Buf<Acc, T, Dim, Idx>;
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;

    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

private:
    Idx arraySize;
    PlatformHost platformHost;
    DevHost devHost;
    PlatformAcc platformAcc;
    DevAcc devAcc;
    BufHost sums;
    BufAcc d_a;
    BufAcc d_b;
    BufAcc d_c;
    BufAcc d_sum;
    Queue queue;
};
