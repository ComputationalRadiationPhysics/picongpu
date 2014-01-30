/* 
 * File:   Buffer.tpp
 * Author: schuma
 *
 * Created on 30. Januar 2014, 13:40
 */

#include "Buffer.hpp"
#include "Environment.hpp"

namespace PMacc
{
    template <class TYPE, unsigned DIM>
    Buffer<TYPE, DIM>::Buffer(DataSpace<DIM> dataSpace) :
    data_space(dataSpace),data1D(true)
    {
        CUDA_CHECK(cudaMallocHost(&current_size, sizeof (size_t)));
        *current_size = dataSpace.getElementCount();
    }

    /**
     * destructor
     */
    template <class TYPE, unsigned DIM>
    Buffer<TYPE, DIM>::~Buffer()
    {
        CUDA_CHECK(cudaFreeHost(current_size));
    }

    /*! Get max spread (elements) of any dimension
     * @return spread (elements) per dimension
     */
    template <class TYPE, unsigned DIM>
    DataSpace<DIM> Buffer<TYPE, DIM>::getDataSpace() const
    {
        return data_space;
    }

    template <class TYPE, unsigned DIM>
    DataSpace<DIM> Buffer<TYPE, DIM>::getCurrentDataSpace() 
    {
        return getCurrentDataSpace(getCurrentSize());
    }

    /*! Spread of memory per dimension which is currently used
     * @return if DIM == DIM1 than return count of elements (x-direction)
     * if DIM == DIM2 than return how many lines (y-direction) of memory is used
     * if DIM == DIM3 than return how many slides (z-direction) of memory is used
     */
    template <class TYPE, unsigned DIM>
    DataSpace<DIM> Buffer<TYPE, DIM>::getCurrentDataSpace(size_t currentSize) 
    {
        DataSpace<DIM> tmp;
        int64_t current_size = static_cast<int64_t>(currentSize);

        //!\todo: current size can be changed if it is a DeviceBuffer and current size is on device
        //call first get current size (but const not allow this)

        if (DIM == DIM1)
        {
            tmp[0] = current_size;
        }
        if (DIM == DIM2)
        {
            if (current_size <= data_space[0])
            {
                tmp[0] = current_size;
                tmp[1] = 1;
            } else
            {
                tmp[0] = data_space[0];
                tmp[1] = (current_size+data_space[0]-1) / data_space[0];
            }
        }
        if (DIM == DIM3)
        {
            if (current_size <= data_space[0])
            {
                tmp[0] = current_size;
                tmp[1] = 1;
                tmp[2] = 1;
            } else if (current_size <= (data_space[0] * data_space[1]))
            {
                tmp[0] = data_space[0];
                tmp[1] = (current_size+data_space[0]-1) / data_space[0];
                tmp[2] = 1;
            } else
            {
                tmp[0] = data_space[0];
                tmp[1] = data_space[1];
                tmp[2] = (current_size+(data_space[0] * data_space[1])-1) / (data_space[0] * data_space[1]);
            }
        }

        return tmp;
    }

    /*! returns the current size (count of elements)
     * @return current size
     */
    template <class TYPE, unsigned DIM>
    size_t Buffer<TYPE, DIM>::getCurrentSize() 
    {
        __startOperation(ITask::TASK_HOST);  
        return *current_size;
    }

    /*! sets the current size (count of elements)
     * @param newsize new current size
     */
    template <class TYPE, unsigned DIM>
    void Buffer<TYPE, DIM>::setCurrentSize(size_t newsize)
    {
        __startOperation(ITask::TASK_HOST); 
        assert(static_cast<size_t>(newsize) <= static_cast<size_t>(data_space.getElementCount()));
        *current_size = newsize;
    }

    template <class TYPE, unsigned DIM>
    inline bool Buffer<TYPE, DIM>::is1D()
    {
        return data1D;
    }
    
    template <class TYPE, unsigned DIM>
    bool Buffer<TYPE, DIM>::isMyDataSpaceGreaterThan(DataSpace<DIM> other)
    {
        return !other.isOneDimensionGreaterThan(data_space);
    }
    
} // namespace PMacc

