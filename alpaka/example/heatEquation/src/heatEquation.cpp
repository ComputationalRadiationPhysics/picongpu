/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file exemplifies usage of alpaka.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>


//#############################################################################
//! alpaka version of explicit finite-difference 1d heat equation solver
//!
//! Solving equation u_t(x, t) = u_xx(x, t) using a simple explicit scheme with
//! forward difference in t and second-order central difference in x
//!
//! \param uCurrBuf grid values of u for each x and the current value of t:
//!                 u(x, t) | t = t_current
//! \param uNext resulting grid values of u for each x and the next value of t:
//!              u(x, t) | t = t_current + dt
//! \param extent number of grid nodes in x (eq. to numNodesX)
//! \param dx step in x
//! \param dt step in t

struct HeatEquationKernel
{
    template<
        typename TAcc
    >
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        double const * const uCurrBuf,
        double * const uNextBuf,
        double const & extent,
        double const & dx,
        double const & dt

    ) const -> void
    {
        // Each kernel executes one element
        double const r = dt / ( dx * dx );
        int idx =
            alpaka::idx::getIdx<
                alpaka::Grid,
                alpaka::Threads
            >( acc )[0];
        if( idx > 0 && idx < extent - 1u )
        {
            uNextBuf[idx] =
                uCurrBuf[idx] * ( 1.0 - 2.0 * r ) +
                uCurrBuf[idx - 1] * r +
                uCurrBuf[idx + 1] * r;
        }
    }
};


//! Exact solution to the test problem
//! u_t(x, t) = u_xx(x, t), x in [0, 1], t in [0, T]
//! u(0, t) = u(1, t) = 0
//! u(x, 0) = sin(pi * x)
//!
//! \param x value of x
//! \param t value of t
double exactSolution(
    double const x,
    double const t
)
{
    constexpr double pi = 3.14159265358979323846;
    return std::exp( -pi * pi * t ) * std::sin( pi * x );
}


//! Each kernel computes the next step for one point.
//! Therefore the number of threads should be equal to numNodesX.
//! Every time step the kernel will be executed numNodesX-times
//! After every step the curr-buffer will be set to the calculated values
//! from the next-buffer.
auto main( ) -> int
{
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#else
    // Parameters (a user is supposed to change numNodesX, numTimeSteps)
    uint32_t const numNodesX = 1000;
    uint32_t const numTimeSteps = 10000;
    double const tMax = 0.001;
    // x in [0, 1], t in [0, tMax]
    double const dx = 1.0 / static_cast< double >( numNodesX - 1 );
    double const dt = tMax / static_cast< double >( numTimeSteps - 1 );

    // Check the stability condition
    double const r = dt / ( dx * dx );
    if( r > 0.5 )
    {
        std::cerr << "Stability condition check failed: dt/dx^2 = " << r
                  << ", it is required to be <= 0.5\n";
        return EXIT_FAILURE;
    }

    // ALPAKA-SETUP

    // Set Dim and Idx type
    using Dim = alpaka::dim::DimInt< 1u >;
    using Idx = uint32_t;

    // Select accelerator-types for host and device
    // using Acc = alpaka::acc::AccCpuSerial<Dim, Idx>;
    using Acc = alpaka::example::ExampleDefaultAcc<Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::acc::getAccName<Acc>() << std::endl;

    using DevHost = alpaka::dev::DevCpu;

    // Select specific devices
    auto const devAcc = alpaka::pltf::getDevByIdx< Acc >( 0u );
    auto const devHost = alpaka::pltf::getDevByIdx< DevHost >( 0u );

    // Get valid workdiv for the given problem
    uint32_t elemPerThread = 1;
    alpaka::vec::Vec<
        Dim,
        Idx
    > const extent { numNodesX };
    using WorkDiv = alpaka::workdiv::WorkDivMembers<
        Dim,
        Idx
    >;
    WorkDiv workdiv {
        alpaka::workdiv::getValidWorkDiv< Acc >(
            devAcc,
            extent,
            elemPerThread,
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted
        )
    };

    // Select queue
    using QueueProperty = alpaka::queue::Blocking;
    using QueueAcc = alpaka::queue::Queue<
        Acc,
        QueueProperty
    >;
    QueueAcc queue { devAcc };

    // Initialize host-buffer
    using BufHost = alpaka::mem::buf::Buf<
        DevHost,
        double,
        Dim,
        Idx
    >;
    // This buffer holds the calculated values
    BufHost uNextBufHost
        {
            alpaka::mem::buf::alloc<
                double,
                Idx
            >(
                devHost,
                extent
            )
        };
    // This buffer will hold the current values (used for the next step)
    BufHost uCurrBufHost
        {
            alpaka::mem::buf::alloc<
                double,
                Idx
            >(
                devHost,
                extent
            )
        };

    double
        * const pCurrHost { alpaka::mem::view::getPtrNative( uCurrBufHost ) };
    double
        * const pNextHost { alpaka::mem::view::getPtrNative( uNextBufHost ) };

    // Accelerator buffer
    using BufAcc = alpaka::mem::buf::Buf<
        Acc,
        double,
        Dim,
        Idx
    >;
    BufAcc uNextBufAcc
        {
            alpaka::mem::buf::alloc<
                double,
                Idx
            >(
                devAcc,
                extent
            )
        };
    BufAcc uCurrBufAcc
        {
            alpaka::mem::buf::alloc<
                double,
                Idx
            >(
                devAcc,
                extent
            )
        };

    double * pCurrAcc { alpaka::mem::view::getPtrNative( uCurrBufAcc ) };
    double * pNextAcc { alpaka::mem::view::getPtrNative( uNextBufAcc ) };

    // Apply initial conditions for the test problem
    for( uint32_t i = 0; i < numNodesX; i++ )
    {
        pCurrHost[i] = exactSolution(
            i * dx,
            0.0
        );
    }

    HeatEquationKernel kernel;

    // Copy host -> device
    alpaka::mem::view::copy(
        queue,
        uCurrBufAcc,
        uCurrBufHost,
        extent
    );

    // EXECUTION
    for( uint32_t step = 0; step < numTimeSteps; step++ )
    {
        // Compute next values
        alpaka::kernel::exec< Acc >(
            queue,
            workdiv,
            kernel,
            pCurrAcc,
            pNextAcc,
            numNodesX,
            dx,
            dt
        );

        // Swap next to curr (shallow copy)
        std::swap(
            pCurrAcc,
            pNextAcc
        );
    }

    // Copy device -> host
    alpaka::mem::view::copy(
        queue,
        uNextBufHost,
        uNextBufAcc,
        extent
    );
    alpaka::wait::wait( queue );

    // Calculate error
    double maxError = 0.0;
    for( uint32_t i = 0; i < numNodesX; i++ )
    {
        auto const error =
            std::abs(
                pNextHost[i] -
                exactSolution(
                    i * dx,
                    tMax
                )
            );
        maxError =
            std::max(
                maxError,
                error
            );
    }
    std::cout << "Max error to the exact solution at t = tMax: " << maxError
              << "\n";

    return EXIT_SUCCESS;
#endif
}
