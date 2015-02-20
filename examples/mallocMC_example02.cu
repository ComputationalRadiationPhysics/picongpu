/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#include <iostream>
#include <assert.h>
#include <vector>
#include <numeric>

#include <cuda.h>
#include <boost/mpl/int.hpp>
#include <boost/mpl/bool.hpp>

///////////////////////////////////////////////////////////////////////////////
// includes for mallocMC
///////////////////////////////////////////////////////////////////////////////
// basic files for mallocMC
#include "src/include/mallocMC/mallocMC_overwrites.hpp"
#include "src/include/mallocMC/mallocMC_hostclass.hpp"

// Load all available policies for mallocMC
#include "src/include/mallocMC/CreationPolicies.hpp"
#include "src/include/mallocMC/DistributionPolicies.hpp"
#include "src/include/mallocMC/OOMPolicies.hpp"
#include "src/include/mallocMC/ReservePoolPolicies.hpp"
#include "src/include/mallocMC/AlignmentPolicies.hpp"
    
///////////////////////////////////////////////////////////////////////////////
// Configuration for mallocMC
///////////////////////////////////////////////////////////////////////////////

// configurate the CreationPolicy "Scatter"
struct ScatterConfig{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;
};

struct ScatterHashParams{
    typedef boost::mpl::int_<38183> hashingK;
    typedef boost::mpl::int_<17497> hashingDistMP;
    typedef boost::mpl::int_<1>     hashingDistWP;
    typedef boost::mpl::int_<1>     hashingDistWPRel;
};

// configure the DistributionPolicy "XMallocSIMD"
struct DistributionConfig{
  typedef ScatterConfig::pagesize pagesize;
};

// configure the AlignmentPolicy "Shrink"
struct AlignmentConfig{
  typedef boost::mpl::int_<16> dataAlignment;
};

// Define a new allocator and call it ScatterAllocator
// which resembles the behaviour of ScatterAlloc
typedef mallocMC::Allocator< 
  mallocMC::CreationPolicies::Scatter<ScatterConfig,ScatterHashParams>,
  mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
  mallocMC::OOMPolicies::ReturnNull,
  mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
  mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>
  > ScatterAllocator;

// use "ScatterAllocator" as mallocMC
MALLOCMC_SET_ALLOCATOR_TYPE(ScatterAllocator)


///////////////////////////////////////////////////////////////////////////////
// End of mallocMC configuration
///////////////////////////////////////////////////////////////////////////////


void run();

int main()
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  if( deviceProp.major < int(2) ) {
    std::cerr << "Error: Compute Capability >= 2.0 required. (is ";
    std::cerr << deviceProp.major << "."<< deviceProp.minor << ")" << std::endl;
    return 1;
  }

  cudaSetDevice(0);
  run();
  cudaDeviceReset();

  return 0;
}


__device__ int** a;
__device__ int** b;
__device__ int** c;


__global__ void createArrays(int x, int y){
  a = (int**) mallocMC::malloc(sizeof(int*) * x*y);
  b = (int**) mallocMC::malloc(sizeof(int*) * x*y);
  c = (int**) mallocMC::malloc(sizeof(int*) * x*y);
}


__global__ void fillArrays(int length, int* d){
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  a[id] = (int*) mallocMC::malloc(sizeof(int)*length);
  b[id] = (int*) mallocMC::malloc(sizeof(int)*length);
  c[id] = (int*) mallocMC::malloc(sizeof(int)*length);
  
  for(int i=0 ; i<length; ++i){
    a[id][i] = id*length+i; 
    b[id][i] = id*length+i;
  }
}


__global__ void addArrays(int length, int* d){
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  d[id] = 0;
  for(int i=0 ; i<length; ++i){
    c[id][i] = a[id][i] + b[id][i];
    d[id] += c[id][i];
  }
}


__global__ void freeArrays(){
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  mallocMC::free(a[id]);
  mallocMC::free(b[id]);
  mallocMC::free(c[id]);
}


void run()
{
  size_t block = 32;
  size_t grid = 32;
  int length = 100;
  assert((unsigned)length <= block*grid); //necessary for used algorithm

  //init the heap
  std::cerr << "initHeap...";
  mallocMC::initHeap(1U*1024U*1024U*1024U); //1GB for device-side malloc
  std::cerr << "done" << std::endl;

  // device-side pointers
  int*  d;
  cudaMalloc((void**) &d, sizeof(int)*block*grid);

  // host-side pointers
  std::vector<int> array_sums(block*grid,0);

  // create arrays of arrays on the device
  createArrays<<<1,1>>>(grid,block);

  // fill 2 of them all with ascending values
  fillArrays<<<grid,block>>>(length, d);

  // add the 2 arrays (vector addition within each thread)
  // and do a thread-wise reduce to d
  addArrays<<<grid,block>>>(length, d);

  cudaMemcpy(&array_sums[0],d,sizeof(int)*block*grid,cudaMemcpyDeviceToHost);

  int sum = std::accumulate(array_sums.begin(),array_sums.end(),0);
  std::cout << "The sum of the arrays on GPU is " << sum << std::endl;

  int n = block*grid*length;
  int gaussian = n*(n-1);
  std::cout << "The gaussian sum as comparison: " << gaussian << std::endl;

  if(mallocMC::Traits<ScatterAllocator>::providesAvailableSlots){
    std::cout << "there are ";
    std::cout << mallocMC::getAvailableSlots(1024U*1024U);
    std::cout << " Slots of size 1MB available" << std::endl;
  }

  freeArrays<<<grid,block>>>();
  cudaFree(d);
  //finalize the heap again
  mallocMC::finalizeHeap();
}
