/* Copyright 2022 Benjamin Worpitz, Bert Wesarg, René Widera, Sergei Bastrakov, Bernhard Manfred Gruber, Mehmet
 * Yusufoglu SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/core/RemoveRestrict.hpp>

#include <tuple>
#include <type_traits>

namespace alpaka
{
    //! \brief The class used to bind kernel function object and arguments together. Once an instance of this class is
    //! created, arguments are not needed to be separately given to functions who need kernel function and arguments.
    //! \tparam TKernelFn The kernel function object type.
    //! \tparam TArgs Kernel function object invocation argument types as a parameter pack.
    template<typename TKernelFn, typename... TArgs>
    class KernelBundle
    {
    public:
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation" // clang does not support the syntax for variadic template
                                                       // arguments "args,...". Ignore the error.
#endif
        //! \param kernelFn The kernel function-object
        //! \param args,... The kernel invocation arguments.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
        KernelBundle(TKernelFn const& kernelFn, TArgs&&... args)
            : m_kernelFn(kernelFn)
            , m_args(std::forward<TArgs>(args)...)
        {
        }

        //! The function object type
        using KernelFn = TKernelFn;
        //! Tuple type to encapsulate kernel function argument types and argument values
        using ArgTuple = std::tuple<remove_restrict_t<std::decay_t<TArgs>>...>;

        KernelFn m_kernelFn;
        ArgTuple m_args;
    };

    //! \brief User defined deduction guide with trailing return type. For CTAD during the construction.
    //! \tparam TKernelFn The kernel function object type.
    //! \tparam TArgs Kernel function object argument types as a parameter pack.
    //! \param kernelFn The kernel object
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation" // clang does not support the syntax for variadic template
                                                       // arguments "args,...". Ignore the error.
#endif
    //! \param args,... The kernel invocation arguments.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    //! \return Kernel function bundle. An instance of KernelBundle which consists the kernel function object and its
    //! arguments.
    template<typename TKernelFn, typename... TArgs>
    KernelBundle(TKernelFn const& kernelFn, TArgs&&... args) -> KernelBundle<TKernelFn, TArgs...>;

} // namespace alpaka
