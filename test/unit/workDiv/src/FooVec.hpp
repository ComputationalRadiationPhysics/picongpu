/* Copyright 2022 Sergei Bastrakov, Jan Stephan, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/dim/Traits.hpp>
#include <alpaka/elem/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

#include <array>
#include <iostream>

//! User defined vector for testing the usability of any vector type.
//!
//! \tparam TVal The data type.
//! \tparam N Vector size as a non-type parameter.
template<typename TVal, std::size_t N>
class FooVec
{
public:
    static_assert(N <= 3, "Size must be 3 or smaller");
    std::array<TVal, N> arr;

    // Default Constructor
    FooVec()
    {
        arr.fill(TVal());
    }

    // Constructor with initializer list
    FooVec(std::initializer_list<TVal> initList)
    {
        if(initList.size() <= N)
        {
            std::copy(initList.begin(), initList.end(), arr.begin());
        }
        else
        {
            throw std::out_of_range("Initializer list size exceeds array size");
        }
    }

    // Example member function to print the contents of the array
    void printArray() const
    {
        for(auto const& element : arr)
        {
            std::cout << element << ' ';
        }
        std::cout << std::endl;
    }
};

namespace alpaka::trait
{

    //! The DimType specialization for the user defined vector
    //! \tparam TVal The data type.
    //! \tparam N Vector size as a non-type parameter.
    template<typename TVal, size_t N>
    struct DimType<FooVec<TVal, N>>
    {
        using type = alpaka::DimInt<N>;
    };

    //! The ElemType specialization for the user defined vector
    //! \tparam TVal The data type.
    //! \tparam N Vector size as a non-type parameter.
    template<typename TVal, size_t N>
    struct ElemType<FooVec<TVal, N>>
    {
        using type = TVal;
    };

    //! The IdxType specialization for the user defined vecto
    //! \tparam TVal The data type.
    //! \tparam N Vector size as a non-type parameter.
    template<typename TVal, size_t N>
    struct IdxType<FooVec<TVal, N>>
    {
        using type = std::size_t;
    };

    //! Specialization for the user defined vector type FooVec. This specialization makes the vector usable in
    //! WorkDivMembers construction. Since alpaka vectors use z-y-x order, FooVec is reversed.
    //! \tparam TVal The element type of the vector type
    //! \tparam N The size of the vector type
    template<typename TVal, size_t N>
    struct GetExtents<FooVec<TVal, N>>
    {
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC constexpr auto operator()(FooVec<TVal, N> const& extent) const
            -> alpaka::Vec<DimInt<N>, TVal>
        {
            alpaka::Vec<DimInt<N>, TVal> v{};
#if BOOST_COMP_NVCC && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 3, 0)
            if(DimInt<N>::value > 0)
#else
            if constexpr(DimInt<N>::value > 0)
#endif
            {
                // Reverse the vector since the dimensions ordered as z-y-x in alpaka
                for(unsigned i = 0; i < DimInt<N>::value; i++)
                    v[i] = extent.arr[DimInt<N>::value - i - 1];
            }

            return v;
        }
    };
} // namespace alpaka::trait
