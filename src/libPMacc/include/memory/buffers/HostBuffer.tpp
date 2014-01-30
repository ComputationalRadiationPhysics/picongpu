/* 
 * File:   HostBuffer.tpp
 * Author: schuma
 *
 * Created on 30. Januar 2014, 13:51
 */

#include "HostBuffer.hpp"

namespace PMacc {

    template <class TYPE, unsigned DIM>
    size_t* HostBuffer<TYPE, DIM>::getCurrentSizePointer() {
        __startOperation(ITask::TASK_HOST);
        return this->current_size;
    }

    template <class TYPE, unsigned DIM>
    HostBuffer<TYPE, DIM>::HostBuffer(DataSpace<DIM> dataSpace) :
    Buffer<TYPE, DIM>(dataSpace) {

    }

    template <class TYPE, unsigned DIM>
    HostBuffer<TYPE, DIM>::~HostBuffer() {
    };
} // namespace PMacc

