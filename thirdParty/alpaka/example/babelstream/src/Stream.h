
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

// NOLINTBEGIN

#pragma once

#include <string>
#include <vector>

// Array values
#define startA (0.1)
#define startB (0.2)
#define startC (0.0)
#define startScalar (0.4)

template<class T>
class Stream
{
public:
    virtual ~Stream()
    {
    }

    // Kernels
    // These must be blocking calls
    virtual void copy() = 0;
    virtual void mul() = 0;
    virtual void add() = 0;
    virtual void triad() = 0;
    virtual void nstream() = 0;
    virtual T dot() = 0;

    // Copy memory between host and device
    virtual void init_arrays(T initA, T initB, T initC) = 0;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) = 0;
};

// Implementation specific device functions
void listDevices(void);
std::string getDeviceName(int const);
std::string getDeviceDriver(int const);

// NOLINTEND
