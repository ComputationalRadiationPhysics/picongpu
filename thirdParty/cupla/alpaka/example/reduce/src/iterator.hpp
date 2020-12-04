/* Copyright 2019 Jonas Schenke
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

#pragma once

#include <alpaka/alpaka.hpp>

//#############################################################################
//! An iterator base class.
//!
//! \tparam T The type.
//! \tparam TBuf The buffer type (standard is T).
template<typename T, typename TBuf = T>
class Iterator
{
protected:
    const TBuf* mData;
    uint64_t mIndex;
    const uint64_t mMaximum;

public:
    //-----------------------------------------------------------------------------
    //! Constructor.
    //!
    //! \param data A pointer to the data.
    //! \param index The index.
    //! \param maximum The first index outside of the iterator memory.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Iterator(const TBuf* data, uint32_t index, uint64_t maximum)
        : mData(data)
        , mIndex(index)
        , mMaximum(maximum)
    {
    }

    //-----------------------------------------------------------------------------
    //! Constructor.
    //!
    //! \param other The other iterator object.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Iterator(const Iterator& other) = default;

    //-----------------------------------------------------------------------------
    //! Compare operator.
    //!
    //! \param other The other object.
    //!
    //! Returns true if objects are equal and false otherwise.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator==(const Iterator& other) const -> bool
    {
        return (this->mData == other.mData) && (this->mIndex == other.mIndex) && (this->mMaximum == other.mMaximum);
    }

    //-----------------------------------------------------------------------------
    //! Compare operator.
    //!
    //! \param other The other object.
    //!
    //! Returns false if objects are equal and true otherwise.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator!=(const Iterator& other) const -> bool
    {
        return !operator==(other);
    }

    //-----------------------------------------------------------------------------
    //! Compare operator.
    //!
    //! \param other The other object.
    //!
    //! Returns false if the other object is equal or smaller and true
    //! otherwise.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator<(const Iterator& other) const -> bool
    {
        return mIndex < other.mIndex;
    }

    //-----------------------------------------------------------------------------
    //! Compare operator.
    //!
    //! \param other The other object.
    //!
    //! Returns false if the other object is equal or bigger and true otherwise.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator>(const Iterator& other) const -> bool
    {
        return mIndex > other.mIndex;
    }

    //-----------------------------------------------------------------------------
    //! Compare operator.
    //!
    //! \param other The other object.
    //!
    //! Returns true if the other object is equal or bigger and false otherwise.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator<=(const Iterator& other) const -> bool
    {
        return mIndex <= other.mIndex;
    }

    //-----------------------------------------------------------------------------
    //! Compare operator.
    //!
    //! \param other The other object.
    //!
    //! Returns true if the other object is equal or smaller and false
    //! otherwise.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator>=(const Iterator& other) const -> bool
    {
        return mIndex >= other.mIndex;
    }

    //-----------------------------------------------------------------------------
    //! Returns the current element.
    //!
    //! Returns a reference to the current index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator*() -> const T&
    {
        return mData[mIndex];
    }
};

//#############################################################################
//! A CPU memory iterator.
//!
//! \tparam TAcc The accelerator type.
//! \tparam T The type.
//! \tparam TBuf The buffer type (standard is T).
template<typename TAcc, typename T, typename TBuf = T>
class IteratorCpu : public Iterator<T, TBuf>
{
public:
    //-----------------------------------------------------------------------------
    //! Constructor.
    //!
    //! \param acc The accelerator object.
    //! \param data A pointer to the data.
    //! \param linearizedIndex The linearized index.
    //! \param gridSize The grid size.
    //! \param n The problem size.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
    IteratorCpu(const TAcc& acc, const TBuf* data, uint32_t linearizedIndex, uint32_t gridSize, uint64_t n)
        : Iterator<T, TBuf>(
            data,
            static_cast<uint32_t>((n * linearizedIndex) / alpaka::math::min(acc, static_cast<uint64_t>(gridSize), n)),
            static_cast<uint32_t>(
                (n * (linearizedIndex + 1)) / alpaka::math::min(acc, static_cast<uint64_t>(gridSize), n)))
    {
    }

    //-----------------------------------------------------------------------------
    //! Returns the iterator for the last item.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto end() const -> IteratorCpu
    {
        IteratorCpu ret(*this);
        ret.mIndex = this->mMaximum;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Increments the internal pointer to the next one and returns this
    //! element.
    //!
    //! Returns a reference to the next index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator++() -> IteratorCpu&
    {
        ++(this->mIndex);
        return *this;
    }

    //-----------------------------------------------------------------------------
    //! Returns the current element and increments the internal pointer to the
    //! next one.
    //!
    //! Returns a reference to the current index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator++(int) -> IteratorCpu
    {
        auto ret(*this);
        ++(this->mIndex);
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Decrements the internal pointer to the previous one and returns the this
    //! element.
    //!
    //! Returns a reference to the previous index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator--() -> IteratorCpu&
    {
        --(this->mIndex);
        return *this;
    }

    //-----------------------------------------------------------------------------
    //! Returns the current element and decrements the internal pointer to the
    //! previous one.
    //!
    //! Returns a reference to the current index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator--(int) -> IteratorCpu
    {
        auto ret(*this);
        --(this->mIndex);
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Returns the index + a supplied offset.
    //!
    //! \param n The offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator+(uint64_t n) const -> IteratorCpu
    {
        IteratorCpu ret(*this);
        ret.mIndex += n;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Returns the index - a supplied offset.
    //!
    //! \param n The offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator-(uint64_t n) const -> IteratorCpu
    {
        IteratorCpu ret(*this);
        ret.mIndex -= n;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Addition assignment.
    //!
    //! \param offset The offset.
    //!
    //! Returns the current object offset by the offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator+=(uint64_t offset) -> IteratorCpu&
    {
        this->mIndex += offset;
        return *this;
    }

    //-----------------------------------------------------------------------------
    //! Substraction assignment.
    //!
    //! \param offset The offset.
    //!
    //! Returns the current object offset by the offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator-=(uint64_t offset) -> IteratorCpu&
    {
        this->mIndex -= offset;
        return *this;
    }
};

//#############################################################################
//! A GPU memory iterator.
//!
//! \tparam TAcc The accelerator type.
//! \tparam T The type.
//! \tparam TBuf The buffer type (standard is T).
template<typename TAcc, typename T, typename TBuf = T>
class IteratorGpu : public Iterator<T, TBuf>
{
private:
    const uint32_t mGridSize;

public:
    //-----------------------------------------------------------------------------
    //! Constructor.
    //!
    //! \param data A pointer to the data.
    //! \param linearizedIndex The linearized index.
    //! \param gridSize The grid size.
    //! \param n The problem size.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
    IteratorGpu(const TAcc&, const TBuf* data, uint32_t linearizedIndex, uint32_t gridSize, uint64_t n)
        : Iterator<T, TBuf>(data, linearizedIndex, n)
        , mGridSize(gridSize)
    {
    }

    //-----------------------------------------------------------------------------
    //! Returns the iterator for the last item.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto end() const -> IteratorGpu
    {
        IteratorGpu ret(*this);
        ret.mIndex = this->mMaximum;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Increments the internal pointer to the next one and returns this
    //! element.
    //!
    //! Returns a reference to the next index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator++() -> IteratorGpu&
    {
        this->mIndex += this->mGridSize;
        return *this;
    }

    //-----------------------------------------------------------------------------
    //! Returns the current element and increments the internal pointer to the
    //! next one.
    //!
    //! Returns a reference to the current index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator++(int) -> IteratorGpu
    {
        auto ret(*this);
        this->mIndex += this->mGridSize;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Decrements the internal pointer to the previous one and returns the this
    //! element.
    //!
    //! Returns a reference to the previous index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator--() -> IteratorGpu&
    {
        this->mIndex -= this->mGridSize;
        return *this;
    }

    //-----------------------------------------------------------------------------
    //! Returns the current element and decrements the internal pointer to the
    //! previous one.
    //!
    //! Returns a reference to the current index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator--(int) -> IteratorGpu
    {
        auto ret(*this);
        this->mIndex -= this->mGridSize;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Returns the index + a supplied offset.
    //!
    //! \param n The offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator+(uint64_t n) const -> IteratorGpu
    {
        auto ret(*this);
        ret.mIndex += n * mGridSize;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Returns the index - a supplied offset.
    //!
    //! \param n The offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator-(uint64_t n) const -> IteratorGpu
    {
        auto ret(*this);
        ret.mIndex -= n * mGridSize;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Addition assignment.
    //!
    //! \param offset The offset.
    //!
    //! Returns the current object offset by the offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator+=(uint64_t offset) -> IteratorGpu&
    {
        this->mIndex += offset * this->mGridSize;
        return *this;
    }

    //-----------------------------------------------------------------------------
    //! Substraction assignment.
    //!
    //! \param offset The offset.
    //!
    //! Returns the current object offset by the offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator-=(uint64_t offset) -> IteratorGpu&
    {
        this->mIndex -= offset * this->mGridSize;
        return *this;
    }
};
