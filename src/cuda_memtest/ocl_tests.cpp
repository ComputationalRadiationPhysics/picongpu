#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>
#include <sys/time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

#include "ocl_tests.h"

#define MIN(x,y) ( (x)<(y)?(x):(y))

unsigned long global_pattern_long = 0;
unsigned int num_iterations=100;
unsigned int num_passes = 0;
__thread char time_buf[128];
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
extern unsigned int exit_on_error;

extern char hostname[];  
extern char* time_string(void);

unsigned int 
error_checking(memtest_control_t* mc)
{
  int i;
  cl_int rc;
  cl_command_queue queue = mc->queue;
  cl_uint host_err_count = 0;
  cl_ulong host_err_addr[MAX_ERR_RECORD_COUNT];
  cl_ulong host_err_expect[MAX_ERR_RECORD_COUNT];
  cl_ulong host_err_current[MAX_ERR_RECORD_COUNT];
  cl_ulong host_err_second_read[MAX_ERR_RECORD_COUNT];

  rc = clEnqueueReadBuffer(queue, mc->err_count, CL_TRUE, 0, sizeof(cl_uint), &host_err_count, 0, NULL, NULL); CLERR;
  rc = clEnqueueReadBuffer(queue, mc->err_addr, CL_TRUE, 0, sizeof(cl_ulong)*MAX_ERR_RECORD_COUNT, host_err_addr, 0, NULL, NULL); CLERR;
  rc = clEnqueueReadBuffer(queue, mc->err_expect, CL_TRUE, 0, sizeof(cl_ulong)*MAX_ERR_RECORD_COUNT, host_err_expect, 0, NULL, NULL); CLERR;
  rc = clEnqueueReadBuffer(queue, mc->err_current, CL_TRUE, 0, sizeof(cl_ulong)*MAX_ERR_RECORD_COUNT, host_err_current, 0, NULL, NULL); CLERR;
  rc = clEnqueueReadBuffer(queue, mc->err_second_read, CL_TRUE, 0, sizeof(cl_ulong)*MAX_ERR_RECORD_COUNT, host_err_second_read, 0, NULL, NULL); CLERR;
  
  if (host_err_count >0){    
    PRINTF("ERROR: error_count=%d\n", host_err_count);
    PRINTF("ERROR: the last %d error addresses are:\t", MIN(MAX_ERR_RECORD_COUNT, host_err_count));
    
    for (i =0;i < MIN(MAX_ERR_RECORD_COUNT, host_err_count); i++){
      printf("%p\t", (void*)host_err_addr[i]);
    }
    printf("\n");
    
    for (i =0; i < MIN(MAX_ERR_RECORD_COUNT, host_err_count); i++){
      PRINTF("ERROR: %dth error, expected value=0x%lx, current value=0x%lx, diff=0x%lx\n",
	     i, host_err_expect[i], host_err_current[i], host_err_expect[i] ^ host_err_current[i]);
      
    }
    
    host_err_count = 0;
    rc = clEnqueueWriteBuffer(queue, mc->err_count, CL_TRUE, 0, sizeof(cl_uint), &host_err_count, 0, NULL, NULL); CLERR;    
    if (exit_on_error){
      PRINTF("Error Found in Memtest, exiting\n");
      exit(1);
    }
  }

  return host_err_count;
}

char*
time_string(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  struct tm tm;
  if (localtime_r(&tv.tv_sec, &tm) == NULL){
        fprintf(stderr, "ERROR: in getting time\n");
        exit(ERR_GENERAL);
  }
  sprintf(time_buf, "%02d/%02d/%04d %02d:%02d:%02d",
	  tm.tm_mon + 1, tm.tm_mday, tm.tm_year + 1900,tm.tm_hour, tm.tm_min, tm.tm_sec);
  
  return time_buf;
}



unsigned long
get_random_num_long(void)
{
    struct timeval t0;
    if (gettimeofday(&t0, NULL) !=0){
      fprintf(stderr, "ERROR: gettimeofday() failed\n");
      exit(ERR_GENERAL);
    }
    
    unsigned int seed= (unsigned int)t0.tv_sec;
    srand(seed);

    unsigned int a = rand_r(&seed);
    unsigned int b = rand_r(&seed);

    unsigned long ret =  ((unsigned long)a) << 32;
    ret |= ((unsigned long)b);

    return ret;
}

void
move_inv_test(memtest_control_t* mc, TYPE p1)
{
  TYPE p2 = ~p1;
  cl_int rc;
  size_t global_work_size[1] = {64*1024};
  size_t local_work_size[1] = {64};
  
  cl_command_queue queue = mc->queue;
  cl_program program = mc->program;
  cl_kernel write_kernel = clCreateKernel(program, "kernel_write", &rc); CLERR;
  cl_kernel read_write_kernel = clCreateKernel(program, "kernel_readwrite", &rc); CLERR;
  cl_kernel read_kernel = clCreateKernel(program, "kernel_read", &rc); CLERR;
  
  rc = clSetKernelArg(write_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(write_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(write_kernel, 2, sizeof(TYPE), &p1); CLERR;
  
  

  rc = clEnqueueNDRangeKernel(queue, write_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL); CLERR;

  rc = clSetKernelArg(read_write_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(read_write_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(read_write_kernel, 2, sizeof(TYPE), &p1); CLERR;
  rc = clSetKernelArg(read_write_kernel, 3, sizeof(TYPE), &p2); CLERR;
  rc = clSetKernelArg(read_write_kernel, 4, sizeof(cl_mem), &mc->err_count); CLERR;
  rc = clSetKernelArg(read_write_kernel, 5, sizeof(cl_mem), &mc->err_addr); CLERR;
  rc = clSetKernelArg(read_write_kernel, 6, sizeof(cl_mem), &mc->err_expect); CLERR;
  rc = clSetKernelArg(read_write_kernel, 7, sizeof(cl_mem), &mc->err_current); CLERR;
  rc = clSetKernelArg(read_write_kernel, 8, sizeof(cl_mem), &mc->err_second_read); CLERR;    
  rc = clEnqueueNDRangeKernel(queue, read_write_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  p1 = p2;
  
  rc = clSetKernelArg(read_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(read_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(read_kernel, 2, sizeof(TYPE), &p1); CLERR;
  rc = clSetKernelArg(read_kernel, 3, sizeof(cl_mem), &mc->err_count); CLERR;
  rc = clSetKernelArg(read_kernel, 4, sizeof(cl_mem), &mc->err_addr); CLERR;
  rc = clSetKernelArg(read_kernel, 5, sizeof(cl_mem), &mc->err_expect); CLERR;
  rc = clSetKernelArg(read_kernel, 6, sizeof(cl_mem), &mc->err_current); CLERR;
  rc = clSetKernelArg(read_kernel, 7, sizeof(cl_mem), &mc->err_second_read); CLERR;
  
  rc = clEnqueueNDRangeKernel(queue, read_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  
  clFinish(queue);
  error_checking(mc);
  clReleaseKernel(write_kernel);
  clReleaseKernel(read_write_kernel);
  clReleaseKernel(read_kernel);
}


/************************************************************************
 * Test0 [Walking 1 bit]
 * This test changes one bit a time in memory address to see it
 * goes to a different memory location. It is designed to test
 * the address wires.
 *
 **************************************************************************/
void
test0(memtest_control_t* mc)
{
  cl_int rc;
  int err_count = 0;
  cl_program program=mc->program;
  cl_command_queue queue  = mc->queue;


  PRINTF("Test0: global walk test\n");
  cl_kernel global_write_kernel = clCreateKernel(program, "kernel0_global_write", &rc); CLERR;
  cl_kernel global_read_kernel = clCreateKernel(program, "kernel0_global_read", &rc); CLERR;
  size_t global_work_size[1] = {1};
  size_t local_work_size[1] = {1};
  rc = clSetKernelArg(global_write_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(global_write_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clEnqueueNDRangeKernel(queue, global_write_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL); CLERR;

  rc = clSetKernelArg(global_read_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(global_read_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(global_read_kernel, 2, sizeof(cl_mem), &mc->err_count); CLERR;
  rc = clSetKernelArg(global_read_kernel, 3, sizeof(cl_mem), &mc->err_addr); CLERR;
  rc = clSetKernelArg(global_read_kernel, 4, sizeof(cl_mem), &mc->err_expect); CLERR;
  rc = clSetKernelArg(global_read_kernel, 5, sizeof(cl_mem), &mc->err_current); CLERR;
  rc = clSetKernelArg(global_read_kernel, 6, sizeof(cl_mem), &mc->err_second_read); CLERR;  
  rc = clEnqueueNDRangeKernel(queue, global_read_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  clFinish(queue);
  err_count = error_checking(mc);
  

  PRINTF("Test0: local walk test\n");  
  cl_kernel local_write_kernel = clCreateKernel(program, "kernel0_local_write", &rc); CLERR;
  cl_kernel local_read_kernel = clCreateKernel(program, "kernel0_local_read", &rc); CLERR;
  global_work_size[0] = 64*1024;
  local_work_size[0] = 64;
  
  rc = clSetKernelArg(local_write_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(local_write_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clEnqueueNDRangeKernel(queue, local_write_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL); CLERR;
  
  rc = clSetKernelArg(local_read_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(local_read_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(local_read_kernel, 2, sizeof(cl_mem), &mc->err_count); CLERR;
  rc = clSetKernelArg(local_read_kernel, 3, sizeof(cl_mem), &mc->err_addr); CLERR;
  rc = clSetKernelArg(local_read_kernel, 4, sizeof(cl_mem), &mc->err_expect); CLERR;
  rc = clSetKernelArg(local_read_kernel, 5, sizeof(cl_mem), &mc->err_current); CLERR;
  rc = clSetKernelArg(local_read_kernel, 6, sizeof(cl_mem), &mc->err_second_read); CLERR;  
  rc = clEnqueueNDRangeKernel(queue, local_read_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  clFinish(queue);
  err_count = error_checking(mc);


  clReleaseKernel(global_write_kernel);
  clReleaseKernel(global_read_kernel);
  clReleaseKernel(local_write_kernel);
  clReleaseKernel(local_read_kernel);
  
}



/*********************************************************************************
 * test1
 * Each Memory location is filled with its own address. The next kernel checks if the
 * value in each memory location still agrees with the address.
 *
 ********************************************************************************/
void
test1(memtest_control_t* mc)
{  
  cl_int rc;
  int err_count = 0;
  cl_program program=mc->program;
  cl_command_queue queue  = mc->queue;
  
  cl_kernel write_kernel = clCreateKernel(program, "kernel1_write", &rc); CLERR;
  cl_kernel read_kernel = clCreateKernel(program, "kernel1_read", &rc); CLERR;
  size_t global_work_size[1] = {64*1024};
  size_t local_work_size[1] = {64};
  
  rc = clSetKernelArg(write_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(write_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clEnqueueNDRangeKernel(queue, write_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL); CLERR;
  
  rc = clSetKernelArg(read_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(read_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(read_kernel, 2, sizeof(cl_mem), &mc->err_count); CLERR;
  rc = clSetKernelArg(read_kernel, 3, sizeof(cl_mem), &mc->err_addr); CLERR;
  rc = clSetKernelArg(read_kernel, 4, sizeof(cl_mem), &mc->err_expect); CLERR;
  rc = clSetKernelArg(read_kernel, 5, sizeof(cl_mem), &mc->err_current); CLERR;
  rc = clSetKernelArg(read_kernel, 6, sizeof(cl_mem), &mc->err_second_read); CLERR;  
  rc = clEnqueueNDRangeKernel(queue, read_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  clFinish(queue);
  err_count = error_checking(mc);
  
  clReleaseKernel(write_kernel);
  clReleaseKernel(read_kernel);
}

/******************************************************************************
 * Test 2 [Moving inversions, ones&zeros]
 * This test uses the moving inversions algorithm with patterns of all
 * ones and zeros.
 *
 ****************************************************************************/
void 
test2(memtest_control_t* mc)
{
  unsigned long p1 = 0;
  unsigned long p2 = ~p1;
  move_inv_test(mc, p1);
  move_inv_test(mc, p2);

}

/*************************************************************************
 *
 * Test 3 [Moving inversions, 8 bit pat]
 * This is the same as test 1 but uses a 8 bit wide pattern of
 * "walking" ones and zeros.  This test will better detect subtle errors
 * in "wide" memory chips. 
 *
 **************************************************************************/
void 
test3(memtest_control_t* mc)
{
  unsigned long p0 = 0x80;
  unsigned long p1 = p0 | (p0<< 8)| (p0<<16)|(p0<<24);
  p1 = p1 | (p1 << 32);
  move_inv_test(mc, p1);
  move_inv_test(mc, ~p1);
}


/************************************************************************************
 * Test 4 [Moving inversions, random pattern]
 * Test 4 uses the same algorithm as test 1 but the data pattern is a
 * random number and it's complement. This test is particularly effective
 * in finding difficult to detect data sensitive errors. A total of 60
 * patterns are used. The random number sequence is different with each pass
 * so multiple passes increase effectiveness.
 *
 *************************************************************************************/

//pretty much the same with test10
void 
test4(memtest_control_t* mc)
{
    int i;
  
  cl_int rc;
  cl_command_queue queue = mc->queue;
  
  TYPE p1;
  p1 = get_random_num_long();
   
  move_inv_test(mc, p1);
}


/************************************************************************************
 * Test 5 [Block move, 64 moves]
 * This test stresses memory by moving block memories. Memory is initialized
 * with shifting patterns that are inverted every 8 bytes.  Then blocks
 * of memory are moved around.  After the moves
 * are completed the data patterns are checked.  Because the data is checked
 * only after the memory moves are completed it is not possible to know
 * where the error occurred.  The addresses reported are only for where the
 * bad pattern was found.
 * 
 *
 *************************************************************************************/


void 
test5(memtest_control_t* mc)
{
  cl_int rc;
  int err_count = 0;
  cl_program program=mc->program;
  cl_command_queue queue  = mc->queue;
  
  cl_kernel init_kernel = clCreateKernel(program, "kernel5_init", &rc); CLERR;
  cl_kernel move_kernel = clCreateKernel(program, "kernel5_move", &rc); CLERR;
  cl_kernel check_kernel = clCreateKernel(program, "kernel5_check", &rc); CLERR;
  size_t global_work_size[1] = {64*1024};
  size_t local_work_size[1] = {64};
  
  rc = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(init_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL); CLERR;
  
  rc = clSetKernelArg(move_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(move_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clEnqueueNDRangeKernel(queue, move_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  
  rc = clSetKernelArg(check_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(check_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(check_kernel, 2, sizeof(cl_mem), &mc->err_count); CLERR;
  rc = clSetKernelArg(check_kernel, 3, sizeof(cl_mem), &mc->err_addr); CLERR;
  rc = clSetKernelArg(check_kernel, 4, sizeof(cl_mem), &mc->err_expect); CLERR;
  rc = clSetKernelArg(check_kernel, 5, sizeof(cl_mem), &mc->err_current); CLERR;
  rc = clSetKernelArg(check_kernel, 6, sizeof(cl_mem), &mc->err_second_read); CLERR;  
  rc = clEnqueueNDRangeKernel(queue, check_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  clFinish(queue);
  err_count = error_checking(mc);
  
  clReleaseKernel(init_kernel);
  clReleaseKernel(move_kernel);
  clReleaseKernel(check_kernel);
  

}





/*****************************************************************************************
 * Test 6 [Moving inversions, 32 bit pat]
 * This is a variation of the moving inversions algorithm that shifts the data
 * pattern left one bit for each successive address. The starting bit position
 * is shifted left for each pass. To use all possible data patterns 32 passes
 * are required.  This test is quite effective at detecting data sensitive
 * errors but the execution time is long.
 *
 ***************************************************************************************/


void
movinv32(memtest_control_t* mc, unsigned int pattern,
         unsigned int lb, unsigned int sval, unsigned int offset)
{
  cl_int rc;
  int err_count = 0;
  cl_program program=mc->program;
  cl_command_queue queue  = mc->queue;
  
  cl_kernel write_kernel = clCreateKernel(program, "kernel_movinv32_write", &rc); CLERR;
  cl_kernel readwrite_kernel = clCreateKernel(program, "kernel_movinv32_readwrite", &rc); CLERR;
  cl_kernel read_kernel = clCreateKernel(program, "kernel_movinv32_read", &rc); CLERR;
  size_t global_work_size[1] = {64*1024};
  size_t local_work_size[1] = {64};
  
  rc = clSetKernelArg(write_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(write_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(write_kernel, 2, sizeof(unsigned int), &pattern); CLERR;
  rc = clSetKernelArg(write_kernel, 3, sizeof(unsigned int), &lb); CLERR;
  rc = clSetKernelArg(write_kernel, 4, sizeof(unsigned int), &sval); CLERR;
  rc = clSetKernelArg(write_kernel, 5, sizeof(unsigned int), &offset); CLERR;
  rc = clEnqueueNDRangeKernel(queue, write_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL); CLERR;
  
  rc = clSetKernelArg(readwrite_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 2, sizeof(unsigned int), &pattern); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 3, sizeof(unsigned int), &lb); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 4, sizeof(unsigned int), &sval); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 5, sizeof(unsigned int), &offset); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 6, sizeof(cl_mem), &mc->err_count); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 7, sizeof(cl_mem), &mc->err_addr); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 8, sizeof(cl_mem), &mc->err_expect); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 9, sizeof(cl_mem), &mc->err_current); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 10, sizeof(cl_mem), &mc->err_second_read); CLERR;  
  rc = clEnqueueNDRangeKernel(queue, readwrite_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  
  rc = clSetKernelArg(read_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(read_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(read_kernel, 2, sizeof(unsigned int), &pattern); CLERR;
  rc = clSetKernelArg(read_kernel, 3, sizeof(unsigned int), &lb); CLERR;
  rc = clSetKernelArg(read_kernel, 4, sizeof(unsigned int), &sval); CLERR;
  rc = clSetKernelArg(read_kernel, 5, sizeof(unsigned int), &offset); CLERR;
  rc = clSetKernelArg(read_kernel, 6, sizeof(cl_mem), &mc->err_count); CLERR;
  rc = clSetKernelArg(read_kernel, 7, sizeof(cl_mem), &mc->err_addr); CLERR;
  rc = clSetKernelArg(read_kernel, 8, sizeof(cl_mem), &mc->err_expect); CLERR;
  rc = clSetKernelArg(read_kernel, 9, sizeof(cl_mem), &mc->err_current); CLERR;
  rc = clSetKernelArg(read_kernel, 10, sizeof(cl_mem), &mc->err_second_read); CLERR;  
  rc = clEnqueueNDRangeKernel(queue, read_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  clFinish(queue);
  err_count = error_checking(mc);
  
  clReleaseKernel(write_kernel);
  clReleaseKernel(readwrite_kernel);
  clReleaseKernel(read_kernel);
  
  return;

}


void 
test6(memtest_control_t* mc)
{
  unsigned int i;
  
  unsigned int pattern;
  
  for (i= 0, pattern = 1;i < 32; pattern = pattern << 1, i++){
    
    PRINTF("Test6[move inversion 32 bits test]: pattern =0x%x, offset=%d\n", pattern, i);
    movinv32(mc, pattern, 1, 0, i);
    PRINTF("Test6[move inversion 32 bits test]: pattern =0x%x, offset=%d\n", ~pattern, i);
    movinv32(mc, ~pattern, 0xfffffffe, 1, i);
    
  }


}


/******************************************************************************
 * Test 7 [Random number sequence]
 *
 * This test writes a series of random numbers into memory.  A block (1 MB) of memory
 * is initialized with random patterns. These patterns and their complements are
 * used in moving inversions test with rest of memory.
 *
 *
 *******************************************************************************/


void 
test7(memtest_control_t* mc)
{
  int i;
  TYPE* host_buf = (TYPE*) malloc(BLOCKSIZE);
  if (host_buf == NULL){
    PRINTF("ERROR: malloc failed for thost buf in test7\n");
    exit(1);
  }
  
  for(i=0;i < BLOCKSIZE/sizeof(TYPE); i++){
    host_buf[i] = get_random_num_long();
  }
  
  cl_int rc = clEnqueueWriteBuffer(mc->queue, mc->device_mem, CL_TRUE, 0, BLOCKSIZE, host_buf, 0, NULL, NULL); CLERR;
  
  int err_count = 0;
  cl_program program=mc->program;
  cl_command_queue queue  = mc->queue;
  
  cl_kernel write_kernel = clCreateKernel(program, "kernel7_write", &rc); CLERR;
  cl_kernel readwrite_kernel = clCreateKernel(program, "kernel7_readwrite", &rc); CLERR;
  cl_kernel read_kernel = clCreateKernel(program, "kernel7_read", &rc); CLERR;
  size_t global_work_size[1] = {64*1024};
  size_t local_work_size[1] = {64};
  
  rc = clSetKernelArg(write_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(write_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clEnqueueNDRangeKernel(queue, write_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL); CLERR;
  
  rc = clSetKernelArg(readwrite_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 2, sizeof(cl_mem), &mc->err_count); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 3, sizeof(cl_mem), &mc->err_addr); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 4, sizeof(cl_mem), &mc->err_expect); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 5, sizeof(cl_mem), &mc->err_current); CLERR;
  rc = clSetKernelArg(readwrite_kernel, 6, sizeof(cl_mem), &mc->err_second_read); CLERR;  
  rc = clEnqueueNDRangeKernel(queue, readwrite_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  
  rc = clSetKernelArg(read_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(read_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(read_kernel, 2, sizeof(cl_mem), &mc->err_count); CLERR;
  rc = clSetKernelArg(read_kernel, 3, sizeof(cl_mem), &mc->err_addr); CLERR;
  rc = clSetKernelArg(read_kernel, 4, sizeof(cl_mem), &mc->err_expect); CLERR;
  rc = clSetKernelArg(read_kernel, 5, sizeof(cl_mem), &mc->err_current); CLERR;
  rc = clSetKernelArg(read_kernel, 6, sizeof(cl_mem), &mc->err_second_read); CLERR;  
  rc = clEnqueueNDRangeKernel(queue, read_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  clFinish(queue);
  err_count = error_checking(mc);
  
  clReleaseKernel(write_kernel);
  clReleaseKernel(readwrite_kernel);
  clReleaseKernel(read_kernel);
  
  free(host_buf);
  
}




/***********************************************************************************
 * Test 8 [Modulo 20, random pattern]
 *
 * A random pattern is generated. This pattern is used to set every 20th memory location
 * in memory. The rest of the memory location is set to the complimemnt of the pattern.
 * Repeat this for 20 times and each time the memory location to set the pattern is shifted right.
 *
 *
 **********************************************************************************/

unsigned int
modtest(memtest_control_t* mc, unsigned int offset,  TYPE p1, TYPE p2)
{
  int err_count = 0;
  cl_program program=mc->program;
  cl_command_queue queue  = mc->queue;
  cl_int rc;
  
  cl_kernel write_kernel = clCreateKernel(program, "kernel_modtest_write", &rc); CLERR;
  cl_kernel read_kernel = clCreateKernel(program, "kernel_modtest_read", &rc); CLERR;
  size_t global_work_size[1] = {64*1024};
  size_t local_work_size[1] = {64};
  
  rc = clSetKernelArg(write_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(write_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(write_kernel, 2, sizeof(unsigned int), &offset); CLERR;  
  rc = clSetKernelArg(write_kernel, 3, sizeof(TYPE), &p1); CLERR;
  rc = clSetKernelArg(write_kernel, 4, sizeof(TYPE), &p2); CLERR;
  rc = clEnqueueNDRangeKernel(queue, write_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL); CLERR;
  
  rc = clSetKernelArg(read_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(read_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(read_kernel, 2, sizeof(unsigned int), &offset); CLERR;  
  rc = clSetKernelArg(read_kernel, 3, sizeof(TYPE), &p1); CLERR;
  rc = clSetKernelArg(read_kernel, 4, sizeof(TYPE), &p2); CLERR;
  rc = clSetKernelArg(read_kernel, 5, sizeof(cl_mem), &mc->err_count); CLERR;
  rc = clSetKernelArg(read_kernel, 6, sizeof(cl_mem), &mc->err_addr); CLERR;
  rc = clSetKernelArg(read_kernel, 7, sizeof(cl_mem), &mc->err_expect); CLERR;
  rc = clSetKernelArg(read_kernel, 8, sizeof(cl_mem), &mc->err_current); CLERR;
  rc = clSetKernelArg(read_kernel, 9, sizeof(cl_mem), &mc->err_second_read); CLERR;
  
  rc = clEnqueueNDRangeKernel(queue, read_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  
  clFinish(queue);
  err_count = error_checking(mc);
  
  clReleaseKernel(write_kernel);
  clReleaseKernel(read_kernel);
  
  
  return err_count;
}

void
test8(memtest_control_t* mc)
{
  unsigned int i;
  unsigned int err= 0;
  unsigned iteration = 0;
  
  TYPE p1;
  TYPE p2;
  if (global_pattern_long){
    p1 = (TYPE)global_pattern_long;
  }else{
    p1 = (TYPE)get_random_num_long();
  }
  
  p2 = ~p1;

 repeat:
  PRINTF("test8[mod test]: p1=0x%lx, p2=0x%lx\n", p1,p2);
   for (i = 0;i < MOD_SZ; i++){
     err += modtest(mc, i, p1, p2);
   }
   
   if (err == 0 && iteration == 0){
     return;
   }
   
   if (iteration < MAX_ERR_ITERATION){
     PRINTF("%dth repeating test8 because there are %d errors found in last run, p1=%lx, p2=%lx\n", iteration, err, p1, p2);
     iteration++;
     err = 0;
     goto repeat;
   }

   return;
}




/************************************************************************************
 *
 * Test 9 [Bit fade test, 90 min, 2 patterns]
 * The bit fade test initializes all of memory with a pattern and then
 * sleeps for 90 minutes. Then memory is examined to see if any memory bits
 * have changed. All ones and all zero patterns are used. This test takes
 * 3 hours to complete.  The Bit Fade test is disabled by default
 *
 **********************************************************************************/
void
test9(memtest_control_t* mc)
{
  cl_program program=mc->program;
  cl_command_queue queue  = mc->queue;
  cl_int rc;
  int sleeptime_in_seconds = 60*90;
  TYPE p1 = 0;
  TYPE p2 = ~p1;
  
  cl_kernel write_kernel = clCreateKernel(program, "kernel_write", &rc); CLERR;
  cl_kernel read_write_kernel = clCreateKernel(program, "kernel_readwrite", &rc); CLERR;
  cl_kernel read_kernel = clCreateKernel(program, "kernel_read", &rc); CLERR;
  size_t global_work_size[1] = {64*1024};
  size_t local_work_size[1] = {64};
  
  rc = clSetKernelArg(write_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(write_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(write_kernel, 2, sizeof(TYPE), &p1); CLERR;
  rc = clEnqueueNDRangeKernel(queue, write_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL); CLERR;
  
  sleep(sleeptime_in_seconds);
  rc = clSetKernelArg(read_write_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(read_write_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(read_write_kernel, 2, sizeof(TYPE), &p1); CLERR;
  rc = clSetKernelArg(read_write_kernel, 3, sizeof(TYPE), &p2); CLERR;
  rc = clSetKernelArg(read_write_kernel, 4, sizeof(cl_mem), &mc->err_count); CLERR;
  rc = clSetKernelArg(read_write_kernel, 5, sizeof(cl_mem), &mc->err_addr); CLERR;
  rc = clSetKernelArg(read_write_kernel, 6, sizeof(cl_mem), &mc->err_expect); CLERR;
  rc = clSetKernelArg(read_write_kernel, 7, sizeof(cl_mem), &mc->err_current); CLERR;
  rc = clSetKernelArg(read_write_kernel, 8, sizeof(cl_mem), &mc->err_second_read); CLERR;
  
  rc = clEnqueueNDRangeKernel(queue, read_write_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  
  sleep(sleeptime_in_seconds);
  clFinish(queue);
  error_checking(mc);
  
  rc = clSetKernelArg(read_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(read_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(read_kernel, 2, sizeof(TYPE), &p2); CLERR;
  rc = clSetKernelArg(read_kernel, 3, sizeof(cl_mem), &mc->err_count); CLERR;
  rc = clSetKernelArg(read_kernel, 4, sizeof(cl_mem), &mc->err_addr); CLERR;
  rc = clSetKernelArg(read_kernel, 5, sizeof(cl_mem), &mc->err_expect); CLERR;
  rc = clSetKernelArg(read_kernel, 6, sizeof(cl_mem), &mc->err_current); CLERR;
  rc = clSetKernelArg(read_kernel, 7, sizeof(cl_mem), &mc->err_second_read); CLERR;
  
  rc = clEnqueueNDRangeKernel(queue, read_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  
  clFinish(queue);
  error_checking(mc);
  clReleaseKernel(write_kernel);
  clReleaseKernel(read_write_kernel);
  clReleaseKernel(read_kernel);
  
}




/**************************************************************************************
 * Test10 [memory stress test]
 *
 * Stress memory as much as we can. A random pattern is generated and a kernel of large grid size
 * and block size is launched to set all memory to the pattern. A new read and write kernel is launched
 * immediately after the previous write kernel to check if there is any errors in memory and set the
 * memory to the compliment. This process is repeated for 1000 times for one pattern. The kernel is
 * written as to achieve the maximum bandwidth between the global memory and GPU.
 * This will increase the chance of catching software error. In practice, we found this test quite useful
 * to flush hardware errors as well.
 * 
 */
  
void test10(memtest_control_t* mc)
{
  int i;
  
  cl_int rc;
  cl_command_queue queue = mc->queue;
  
  TYPE p1;
  if (global_pattern_long){
    p1 = global_pattern_long;
  }else{
    p1 = get_random_num_long();
  }
  
  TYPE p2 = ~p1;
  int n = num_iterations;
  size_t global_work_size[1] = {64*1024};
  size_t local_work_size[1] = {64};
  
  cl_program program = mc->program;  
  cl_kernel write_kernel = clCreateKernel(program, "kernel_write", &rc); CLERR;
  cl_kernel read_write_kernel = clCreateKernel(program, "kernel_readwrite", &rc); CLERR;
  cl_kernel read_kernel = clCreateKernel(program, "kernel_read", &rc); CLERR;
  
  rc = clSetKernelArg(write_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(write_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(write_kernel, 2, sizeof(TYPE), &p1); CLERR;
  
  

  rc = clEnqueueNDRangeKernel(queue, write_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL); CLERR;

  for(i=0;i < n; i++){
    rc = clSetKernelArg(read_write_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
    rc = clSetKernelArg(read_write_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
    rc = clSetKernelArg(read_write_kernel, 2, sizeof(TYPE), &p1); CLERR;
    rc = clSetKernelArg(read_write_kernel, 3, sizeof(TYPE), &p2); CLERR;
    rc = clSetKernelArg(read_write_kernel, 4, sizeof(cl_mem), &mc->err_count); CLERR;
    rc = clSetKernelArg(read_write_kernel, 5, sizeof(cl_mem), &mc->err_addr); CLERR;
    rc = clSetKernelArg(read_write_kernel, 6, sizeof(cl_mem), &mc->err_expect); CLERR;
    rc = clSetKernelArg(read_write_kernel, 7, sizeof(cl_mem), &mc->err_current); CLERR;
    rc = clSetKernelArg(read_write_kernel, 8, sizeof(cl_mem), &mc->err_second_read); CLERR;    
    rc = clEnqueueNDRangeKernel(queue, read_write_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
    p1 = p2;
    p2 = ~p1;
  }
  
  rc = clSetKernelArg(read_kernel, 0, sizeof(cl_mem), &mc->device_mem); CLERR;
  rc = clSetKernelArg(read_kernel, 1, sizeof(cl_ulong), &mc->mem_size); CLERR;
  rc = clSetKernelArg(read_kernel, 2, sizeof(TYPE), &p1); CLERR;
  rc = clSetKernelArg(read_kernel, 3, sizeof(cl_mem), &mc->err_count); CLERR;
  rc = clSetKernelArg(read_kernel, 4, sizeof(cl_mem), &mc->err_addr); CLERR;
  rc = clSetKernelArg(read_kernel, 5, sizeof(cl_mem), &mc->err_expect); CLERR;
  rc = clSetKernelArg(read_kernel, 6, sizeof(cl_mem), &mc->err_current); CLERR;
  rc = clSetKernelArg(read_kernel, 7, sizeof(cl_mem), &mc->err_second_read); CLERR;
  
  rc = clEnqueueNDRangeKernel(queue, read_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);CLERR;
  
  clFinish(queue);
  error_checking(mc);
  clReleaseKernel(write_kernel);
  clReleaseKernel(read_write_kernel);
  clReleaseKernel(read_kernel);
  return;
  
}


cuda_memtest_t cuda_memtests[]={
  {test0, "Test0 [Walking 1 bit]",                    1},
  {test1, "Test1 [Own address test]",                 1},
  {test2, "Test2 [Moving inversions, ones&zeros]",    1},
  {test3, "Test3 [Moving inversions, 8 bit pat]",     1},
  {test4, "Test4 [Moving inversions, random pattern]",1},
  {test5, "Test5 [Block move, 64 moves]",             1},
  {test6, "Test6 [Moving inversions, 32 bit pat]",    1},
  {test7, "Test7 [Random number sequence]",           1},
  {test8, "Test8 [Modulo 20, random pattern]",        1},
  {test9, "Test9 [Bit fade test]",                    0},
  {test10, "Test10 [Memory stress test]",             1},
  
};



void* run_tests(void* arg)
{
  memtest_control_t* mc = (memtest_control_t*)arg;
  struct timeval t0, t1;
  unsigned int pass = 0;
  int i;
  while(1){   
    for (i = 0;i < DIM(cuda_memtests); i++){
      if (cuda_memtests[i].enabled){
	PRINTF("%s\n", cuda_memtests[i].desc);
	gettimeofday(&t0, NULL);
	cuda_memtests[i].func(mc);
	gettimeofday(&t1, NULL);
	PRINTF("Test%d finished in %.1f seconds\n", i, TDIFF(t1, t0));
      }//if
    }//for
        

    if (num_passes <=0){
      continue;
    }

    pass++;
    if (pass >= num_passes){
      break;
    }
  }  
  return NULL;
  }
