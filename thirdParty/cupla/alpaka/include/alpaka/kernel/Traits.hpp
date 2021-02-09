/* Copyright 2019-2020 Axel Huebl, Benjamin Worpitz, René Widera, Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Debug.hpp>
#include <alpaka/core/OmpSchedule.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/meta/Void.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
#    include <alpaka/workdiv/Traits.hpp>
#endif

#include <type_traits>

//-----------------------------------------------------------------------------
//! The alpaka accelerator library.
namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The kernel traits.
    namespace traits
    {
        //#############################################################################
        //! The kernel execution task creation trait.
        template<
            typename TAcc,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs/*,
            typename TSfinae = void*/>
        struct CreateTaskKernel;

        //#############################################################################
        //! The trait for getting the size of the block shared dynamic memory of a kernel.
        //!
        //! \tparam TKernelFnObj The kernel function object.
        //! \tparam TAcc The accelerator.
        //!
        //! The default implementation returns 0.
        template<typename TKernelFnObj, typename TAcc, typename TSfinae = void>
        struct BlockSharedMemDynSizeBytes
        {
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif
            //-----------------------------------------------------------------------------
            //! \param kernelFnObj The kernel object for which the block shared memory size should be calculated.
            //! \param blockThreadExtent The block thread extent.
            //! \param threadElemExtent The thread element extent.
            //! \tparam TArgs The kernel invocation argument types pack.
            //! \param args,... The kernel invocation arguments.
            //! \return The size of the shared memory allocated for a block in bytes.
            //! The default version always returns zero.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TDim, typename... TArgs>
            ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
                TKernelFnObj const& kernelFnObj,
                Vec<TDim, Idx<TAcc>> const& blockThreadExtent,
                Vec<TDim, Idx<TAcc>> const& threadElemExtent,
                TArgs const&... args) -> std::size_t
            {
                alpaka::ignore_unused(kernelFnObj);
                alpaka::ignore_unused(blockThreadExtent);
                alpaka::ignore_unused(threadElemExtent);
                alpaka::ignore_unused(args...);

                return 0u;
            }
        };

        namespace detail
        {
            //#############################################################################
            //! Functor to get OpenMP schedule defined by kernel class.
            //! When no schedule is defined, return a default one.
            template<class TKernel, class = void>
            struct GetOmpSchedule
            {
                ALPAKA_FN_HOST static auto get()
                {
                    return alpaka::omp::Schedule{};
                }
            };

            //! Functor to get OpenMP schedule for kernel classes with
            //! ompSchedule static member.
            //! That member is never odr-used by alpaka.
            template<class TKernel>
            struct GetOmpSchedule<TKernel, meta::Void<decltype(TKernel::ompSchedule)>>
            {
                ALPAKA_FN_HOST static auto get()
                {
                    // Just having return TKernel::ompSchedule here would be
                    // a non-odr use of that variable, since it would be an
                    // argument of the copy constructor. So have to manually
                    // create a new identical object and then return it.
                    return alpaka::omp::Schedule{TKernel::ompSchedule.kind, TKernel::ompSchedule.chunkSize};
                }
            };
        } // namespace detail

        //#############################################################################
        //! The trait for getting the schedule to use when a kernel is run using
        //! the CpuOmp2Blocks accelerator.
        //!
        //! Has no effect on other accelerators or when run using OpenMP
        //! implementation not supporting at least version 3.0.
        //!
        //! A user could either specialize this trait for their kernel, or define
        //! a public static member ompSchedule of type alpaka::omp::Schedule
        //! inside it, which would be picked up by this implementation.
        //! In the latter case, alpaka never odr-uses that member.
        //!
        //! \tparam TKernelFnObj The kernel function object.
        //! \tparam TAcc The accelerator.
        //!
        //! The default implementation returns 0.
        template<typename TKernelFnObj, typename TAcc, typename TSfinae = void>
        struct OmpSchedule
        {
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif
            //-----------------------------------------------------------------------------
            //! \param kernelFnObj The kernel object for which the schedule should be returned.
            //! \param blockThreadExtent The block thread extent.
            //! \param threadElemExtent The thread element extent.
            //! \tparam TArgs The kernel invocation argument types pack.
            //! \param args,... The kernel invocation arguments.
            //! \return The OpenMP schedule information.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TDim, typename... TArgs>
            ALPAKA_FN_HOST static auto getOmpSchedule(
                TKernelFnObj const& kernelFnObj,
                Vec<TDim, Idx<TAcc>> const& blockThreadExtent,
                Vec<TDim, Idx<TAcc>> const& threadElemExtent,
                TArgs const&... args) -> alpaka::omp::Schedule
            {
                alpaka::ignore_unused(kernelFnObj);
                alpaka::ignore_unused(blockThreadExtent);
                alpaka::ignore_unused(threadElemExtent);
                alpaka::ignore_unused(args...);

                return detail::GetOmpSchedule<TKernelFnObj>::get();
            }
        };
    } // namespace traits

#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif
    //-----------------------------------------------------------------------------
    //! \tparam TAcc The accelerator type.
    //! \param kernelFnObj The kernel object for which the block shared memory size should be calculated.
    //! \param blockThreadExtent The block thread extent.
    //! \param threadElemExtent The thread element extent.
    //! \param args,... The kernel invocation arguments.
    //! \return The size of the shared memory allocated for a block in bytes.
    //! The default implementation always returns zero.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TKernelFnObj, typename TDim, typename... TArgs>
    ALPAKA_FN_HOST_ACC auto getBlockSharedMemDynSizeBytes(
        TKernelFnObj const& kernelFnObj,
        Vec<TDim, Idx<TAcc>> const& blockThreadExtent,
        Vec<TDim, Idx<TAcc>> const& threadElemExtent,
        TArgs const&... args) -> std::size_t
    {
        return traits::BlockSharedMemDynSizeBytes<TKernelFnObj, TAcc>::getBlockSharedMemDynSizeBytes(
            kernelFnObj,
            blockThreadExtent,
            threadElemExtent,
            args...);
    }

#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif
    //-----------------------------------------------------------------------------
    //! \tparam TAcc The accelerator type.
    //! \param kernelFnObj The kernel object for which the block shared memory size should be calculated.
    //! \param blockThreadExtent The block thread extent.
    //! \param threadElemExtent The thread element extent.
    //! \param args,... The kernel invocation arguments.
    //! \return The OpenMP schedule information.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    template<typename TAcc, typename TKernelFnObj, typename TDim, typename... TArgs>
    ALPAKA_FN_HOST auto getOmpSchedule(
        TKernelFnObj const& kernelFnObj,
        Vec<TDim, Idx<TAcc>> const& blockThreadExtent,
        Vec<TDim, Idx<TAcc>> const& threadElemExtent,
        TArgs const&... args) -> omp::Schedule
    {
        return traits::OmpSchedule<TKernelFnObj, TAcc>::getOmpSchedule(
            kernelFnObj,
            blockThreadExtent,
            threadElemExtent,
            args...);
    }

#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif

    namespace detail
    {
        //#############################################################################
        //! Check that the return of TKernelFnObj is void
        template<typename TAcc, typename TSfinae = void>
        struct CheckFnReturnType
        {
            template<typename TKernelFnObj, typename... TArgs>
            void operator()(TKernelFnObj const&, TArgs const&...)
            {
#if defined(__cpp_lib_is_invocable) && __cpp_lib_is_invocable >= 201703
                using Result = std::invoke_result_t<TKernelFnObj, TAcc const&, TArgs const&...>;
#else
                using Result = std::result_of_t<TKernelFnObj(TAcc const&, TArgs const&...)>;
#endif
                static_assert(std::is_same<Result, void>::value, "The TKernelFnObj is required to return void!");
            }
        };
    } // namespace detail
    //-----------------------------------------------------------------------------
    //! Creates a kernel execution task.
    //!
    //! \tparam TAcc The accelerator type.
    //! \param workDiv The index domain work division.
    //! \param kernelFnObj The kernel function object which should be executed.
    //! \param args,... The kernel invocation arguments.
    //! \return The kernel execution task.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    template<typename TAcc, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
    ALPAKA_FN_HOST auto createTaskKernel(TWorkDiv const& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
    {
        // check for void return type
        detail::CheckFnReturnType<TAcc>{}(kernelFnObj, args...);

        static_assert(
            Dim<std::decay_t<TWorkDiv>>::value == Dim<TAcc>::value,
            "The dimensions of TAcc and TWorkDiv have to be identical!");
        static_assert(
            std::is_same<Idx<std::decay_t<TWorkDiv>>, Idx<TAcc>>::value,
            "The idx type of TAcc and the idx type of TWorkDiv have to be identical!");

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
        std::cout << __func__ << " workDiv: " << workDiv << ", kernelFnObj: " << typeid(kernelFnObj).name()
                  << std::endl;
#endif
        return traits::CreateTaskKernel<TAcc, TWorkDiv, TKernelFnObj, TArgs...>::createTaskKernel(
            workDiv,
            kernelFnObj,
            std::forward<TArgs>(args)...);
    }

#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif
    //-----------------------------------------------------------------------------
    //! Executes the given kernel in the given queue.
    //!
    //! \tparam TAcc The accelerator type.
    //! \param queue The queue to enqueue the view copy task into.
    //! \param workDiv The index domain work division.
    //! \param kernelFnObj The kernel function object which should be executed.
    //! \param args,... The kernel invocation arguments.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    template<typename TAcc, typename TQueue, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
    ALPAKA_FN_HOST auto exec(TQueue& queue, TWorkDiv const& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
        -> void
    {
        enqueue(queue, createTaskKernel<TAcc>(workDiv, kernelFnObj, std::forward<TArgs>(args)...));
    }
} // namespace alpaka
