/*
 * Illinois Open Source License
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright © 2009,    University of Illinois.  All rights reserved.
 *
 * Developed by:
 *
 * Innovative Systems Lab
 * National Center for Supercomputing Applications
 * http://www.ncsa.uiuc.edu/AboutUs/Directorates/ISL.html
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal with
 * the Software without restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
 * Software, and to permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * * Redistributions of source code must retain the above copyright notice, this list
 * of conditions and the following disclaimers.
 *
 * * Redistributions in binary form must reproduce the above copyright notice, this list
 * of conditions and the following disclaimers in the documentation and/or other materials
 * provided with the distribution.
 *
 * * Neither the names of the Innovative Systems Lab, the National Center for Supercomputing
 * Applications, nor the names of its contributors may be used to endorse or promote products
 * derived from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
 * OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 */

#include <stdio.h>
#include "misc.h"
#include <cuda.h>
#include <sys/time.h>
#include <unistd.h>

static __thread unsigned long* err_addr;
static __thread unsigned long* err_expect;
static __thread unsigned long* err_current;
static __thread unsigned long* err_second_read;
static __thread unsigned int* err_count;
static __thread unsigned int unreported_errors=0;
__thread struct timeval last_report_time;
extern unsigned int report_interval;
__thread unsigned int firsttime=1;

__thread char time_buf[128];
extern unsigned exit_on_error ;
extern unsigned int email_notification;
extern char emails[];
extern unsigned int global_pattern;
extern unsigned long global_pattern_long;
extern unsigned int num_iterations;
extern unsigned int num_passes;
extern char driver_info[MAX_STR_LEN];

#define MAX_ERR_RECORD_COUNT 10

#ifdef SM_10
#define atomicAdd(x, y) do{ (*x) = (*x) + y ;}while(0)
#define RECORD_ERR(err, p, expect, current) do{	  \
	atomicAdd(err, 1); \
	}while(0)
#else

#define RECORD_ERR(err, p, expect, current) do{		\
	unsigned int idx = atomicAdd(err, 1);		\
	idx = idx % MAX_ERR_RECORD_COUNT;		\
	err_addr[idx] = (unsigned long)p;		\
	err_expect[idx] = (unsigned long)expect;	\
	err_current[idx] = (unsigned long)current;	\
	err_second_read[idx] = (unsigned long)(*p);	\
}while(0)


#endif

#define MAX_ITERATION 3

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


unsigned int
error_checking(const char* msg, unsigned int blockidx)
{
    unsigned int err = 0;
    unsigned long host_err_addr[MAX_ERR_RECORD_COUNT];
    unsigned long host_err_expect[MAX_ERR_RECORD_COUNT];
    unsigned long host_err_current[MAX_ERR_RECORD_COUNT];
    unsigned long host_err_second_read[MAX_ERR_RECORD_COUNT];
    unsigned int i;

    cudaMemcpy((void*)&err, (void*)err_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);CUERR;
    cudaMemcpy((void*)&host_err_addr[0], (void*)err_addr, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT, cudaMemcpyDeviceToHost);CUERR;
    cudaMemcpy((void*)&host_err_expect[0], (void*)err_expect, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT, cudaMemcpyDeviceToHost);CUERR;
    cudaMemcpy((void*)&host_err_current[0], (void*)err_current, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT, cudaMemcpyDeviceToHost);CUERR;
    cudaMemcpy((void*)&host_err_second_read[0], (void*)err_second_read, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT, cudaMemcpyDeviceToHost);CUERR;

#define ERR_MSG_LENGTH 4096
    char error_msg[ERR_MSG_LENGTH];
    char* emsg = error_msg;
    if (err){
	emsg += sprintf(emsg, "Unreported errors since last email: %d\n", unreported_errors);

	FPRINTF("ERROR: %s",  driver_info);
	emsg += sprintf(emsg, "ERROR: %s", driver_info);

#if !defined(NVML_DEVICE_SERIAL_BUFFER_SIZE)
	char devSerialNum[] = "unknown (no NVML found)";
#else
	char devSerialNum[NVML_DEVICE_SERIAL_BUFFER_SIZE];
	get_serial_number( gpu_idx, devSerialNum );
#endif
	FPRINTF("ERROR: The unit serial number is %s\n", devSerialNum);
	emsg += sprintf(emsg, "ERROR: The unit serial number is %s\n", devSerialNum);

	FPRINTF("ERROR: (%s) %d errors found in block %d\n", msg, err, blockidx);
	emsg += sprintf(emsg, "ERROR: (%s) %d errors found in block %d\n", msg, err, blockidx);

	FPRINTF("ERROR: the last %d error addresses are:\t", MIN(MAX_ERR_RECORD_COUNT, err));
	emsg += sprintf(emsg, "ERROR: the last %d error addresses are:\t", MIN(MAX_ERR_RECORD_COUNT, err));

	for (i =0;i < MIN(MAX_ERR_RECORD_COUNT, err); i++){
	    fprintf(stderr, "%p\t", (void*)host_err_addr[i]);
	    emsg += sprintf(emsg, "%p\t", (void*)host_err_addr[i]);
	}
	fprintf(stderr, "\n");
	emsg += sprintf(emsg, "\n");

	for (i =0; i < MIN(MAX_ERR_RECORD_COUNT, err); i++){
	    FPRINTF("ERROR: %dth error, expected value=0x%lx, current value=0x%lx, diff=0x%lx (second_read=0x%lx, expect=0x%lx, diff with expected value=0x%lx)\n",
		    i, host_err_expect[i], host_err_current[i], host_err_expect[i] ^ host_err_current[i],
		    host_err_second_read[i], host_err_expect[i]  , host_err_expect[i] ^ host_err_second_read[i]);
	    emsg += sprintf(emsg, "ERROR: %dth error, expected value=0x%lx, current value=0x%lx, diff=0x%lx (second_read=0x%lx, expect=0x%lx, diff with expected value=0x%lx)\n",
			    i, host_err_expect[i], host_err_current[i], host_err_expect[i] ^ host_err_current[i],
			    host_err_second_read[i], host_err_expect[i], host_err_expect[i] ^ host_err_second_read[i]);


	}

	if (email_notification){

	    struct timeval tv;
	    gettimeofday(&tv, NULL);
	    if ( firsttime || TDIFF(tv, last_report_time) > report_interval) {

		FPRINTF("ERROR: reporting this error to %s\n", emails);

#define CMD_LENGTH (ERR_MSG_LENGTH + 256)
		char cmd[CMD_LENGTH];
		error_msg[ERR_MSG_LENGTH -1] = 0;
		snprintf(cmd, CMD_LENGTH, "echo \"%s cuda_memtest errors found in %s[%d]\n%s\" |%s -s \" cuda_memtest errors found in %s[%d]\" %s",
			 time_string(), hostname,gpu_idx, error_msg, MAILFILE, hostname,gpu_idx, emails );
		system(cmd);

		firsttime = 0;
		unreported_errors = 0;
		last_report_time = tv;

	    }else{
		FPRINTF("ERROR: this error is not email reported\n");
		unreported_errors ++;
	    }
	}
	cudaMemset(err_count, 0, sizeof(unsigned int));CUERR;
	cudaMemset((void*)&err_addr[0], 0, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT);CUERR;
	cudaMemset((void*)&err_expect[0], 0, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT);CUERR;
	cudaMemset((void*)&err_current[0], 0, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT);CUERR;
	if (exit_on_error){
	    cudaDeviceReset();
	    exit(ERR_BAD_STATE);
	}
    }

    return err;
}

unsigned int
get_random_num(void)
{
    struct timeval t0;
    if (gettimeofday(&t0, NULL) !=0){
	fprintf(stderr, "ERROR: gettimeofday() failed\n");
	exit(ERR_GENERAL);
    }

    unsigned int seed= (unsigned int)t0.tv_sec;
    srand(seed);

    return rand_r(&seed);
}

uint64_t
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

    uint64_t ret =  ((uint64_t)a) << 32;
    ret |= ((uint64_t)b);

    return ret;
}





__global__ void
kernel_move_inv_write(char* _ptr, char* end_ptr, unsigned int pattern)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	ptr[i] = pattern;
    }

    return;
}


__global__ void
kernel_move_inv_readwrite(char* _ptr, char* end_ptr, unsigned int p1, unsigned int p2, unsigned int* err,
			  unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (ptr[i] != p1){
	    RECORD_ERR(err, &ptr[i], p1, ptr[i]);
	}
	ptr[i] = p2;

    }

    return;
}


__global__ void
kernel_move_inv_read(char* _ptr, char* end_ptr,  unsigned int pattern, unsigned int* err,
		     unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read )
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (ptr[i] != pattern){
	    RECORD_ERR(err, &ptr[i], pattern, ptr[i]);
	}
    }

    return;
}


unsigned int
move_inv_test(char* ptr, unsigned int tot_num_blocks, unsigned int p1, unsigned p2)
{

    unsigned int i;
    unsigned int err = 0;
    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;

    for (i= 0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_move_inv_write<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, p1); SYNC_CUERR;
	SHOW_PROGRESS("move_inv_write", i, tot_num_blocks);
    }


    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_move_inv_readwrite<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, p1, p2, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
	err += error_checking("move_inv_readwrite",  i);
	SHOW_PROGRESS("move_inv_readwrite", i, tot_num_blocks);
    }

    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_move_inv_read<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, p2, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
	err += error_checking("move_inv_read",  i);
	SHOW_PROGRESS("move_inv_read", i, tot_num_blocks);
    }

    return err;

}


/*
 * Test0 [Walking 1 bit]
 * This test changes one bit a time in memory address to see it
 * goes to a different memory location. It is designed to test
 * the address wires.
 */



/*
__global__ void
kernel_test0_write(char* _ptr, char* end_ptr, unsigned int pattern,
		   unsigned int* err, unsigned long* err_addr,
		   unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }


    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	ptr[i] = pattern;
    }

    return;
}


__global__ void
kernel_test0_readwrite(char* _ptr, char* end_ptr, unsigned int pattern, unsigned int prev_pattern,
	     unsigned int* err, unsigned long* err_addr,
	     unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }


    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (ptr[i] != prev_pattern){
	    RECORD_ERR(err, &ptr[i], prev_pattern, ptr[i]);
	}
	ptr[i] = pattern;
    }

    return;
}


void
test0(char* ptr, unsigned int tot_num_blocks)
{

    unsigned int i,j;
    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;

    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	unsigned int prev_pattern = 0;

	//the first iteration
	unsigned int pattern = 1;
	kernel_test0_write<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, pattern,  err_count, err_addr, err_expect, err_current, err_second_read); CUERR;
	prev_pattern =pattern;

	for (j =1; j < 32; j++){
	    pattern = 1 << j;
	    kernel_test0_readwrite<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, pattern, prev_pattern, err_count, err_addr, err_expect, err_current, err_second_read); CUERR;
	    prev_pattern = pattern;
	}

	error_checking(__FUNCTION__,  i);
	SHOW_PROGRESS(__FUNCTION__, i, tot_num_blocks);
    }


    return;

}
*/

__global__ void
kernel_test0_global_write(char* _ptr, char* _end_ptr)
{

    unsigned int* ptr = (unsigned int*)_ptr;
    unsigned int* end_ptr = (unsigned int*)_end_ptr;
    unsigned int* orig_ptr = ptr;

    unsigned int pattern = 1;

    unsigned long mask = 4;

    *ptr = pattern;

    while(ptr < end_ptr){

	ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);
	if (ptr == orig_ptr){
	    mask = mask <<1;
	    continue;
	}
	if (ptr >= end_ptr){
	    break;
	}

	*ptr = pattern;

	pattern = pattern << 1;
	mask = mask << 1;
    }
    return;
}

__global__ void
kernel_test0_global_read(char* _ptr, char* _end_ptr, unsigned int* err, unsigned long* err_addr,
			 unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int* ptr = (unsigned int*)_ptr;
    unsigned int* end_ptr = (unsigned int*)_end_ptr;
    unsigned int* orig_ptr = ptr;

    unsigned int pattern = 1;

    unsigned long mask = 4;

    if (*ptr != pattern){
	RECORD_ERR(err, ptr, pattern, *ptr);
    }

    while(ptr < end_ptr){

	ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);
	if (ptr == orig_ptr){
	    mask = mask <<1;
	    continue;
	}
	if (ptr >= end_ptr){
	    break;
	}

	if (*ptr != pattern){
	    RECORD_ERR(err, ptr, pattern, *ptr);
	}

	pattern = pattern << 1;
	mask = mask << 1;
    }
    return;
}



__global__ void
kernel_test0_write(char* _ptr, char* end_ptr)
{

    unsigned int* orig_ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);;
    unsigned int* ptr = orig_ptr;
    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    unsigned int* block_end = orig_ptr + BLOCKSIZE/sizeof(unsigned int);

    unsigned int pattern = 1;

    unsigned long mask = 4;

    *ptr = pattern;

    while(ptr < block_end){

	ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);
	if (ptr == orig_ptr){
	    mask = mask <<1;
	    continue;
	}
	if (ptr >= block_end){
	    break;
	}

	*ptr = pattern;

	pattern = pattern << 1;
	mask = mask << 1;
    }
    return;
}


__global__ void
kernel_test0_read(char* _ptr, char* end_ptr, unsigned int* err, unsigned long* err_addr,
		  unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{

    unsigned int* orig_ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);;
    unsigned int* ptr = orig_ptr;
    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    unsigned int* block_end = orig_ptr + BLOCKSIZE/sizeof(unsigned int);

    unsigned int pattern = 1;

    unsigned long mask = 4;
    if (*ptr != pattern){
	RECORD_ERR(err, ptr, pattern, *ptr);
    }

    while(ptr < block_end){

	ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);
	if (ptr == orig_ptr){
	    mask = mask <<1;
	    continue;
	}
	if (ptr >= block_end){
	    break;
	}

	if (*ptr != pattern){
	    RECORD_ERR(err, ptr, pattern, *ptr);
	}

	pattern = pattern << 1;
	mask = mask << 1;
    }
    return;
}


void
test0(char* ptr, unsigned int tot_num_blocks)
{

    unsigned int i;
    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;


    //test global address
    kernel_test0_global_write<<<1, 1>>>(ptr, end_ptr); SYNC_CUERR;
    kernel_test0_global_read<<<1, 1>>>(ptr, end_ptr, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
    error_checking("test0 on global address",  0);

    for(unsigned int ite = 0;ite < num_iterations; ite++){
	for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	    dim3 grid;
	    grid.x= GRIDSIZE;
	    kernel_test0_write<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr); SYNC_CUERR;
	    SHOW_PROGRESS("test0 on writing", i, tot_num_blocks);
	}

	for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	    dim3 grid;
	    grid.x= GRIDSIZE;
	    kernel_test0_read<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
	    error_checking(__FUNCTION__,  i);
	    SHOW_PROGRESS("test0 on reading", i, tot_num_blocks);
	}
    }
    return;

}



/*********************************************************************************
 * test1
 * Each Memory location is filled with its own address. The next kernel checks if the
 * value in each memory location still agrees with the address.
 *
 ********************************************************************************/

__global__ void
kernel_test1_write(char* _ptr, char* end_ptr, unsigned int* err)
{
    unsigned int i;
    unsigned long* ptr = (unsigned long*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned long*) end_ptr) {
	return;
    }


    for (i = 0;i < BLOCKSIZE/sizeof(unsigned long); i++){
	ptr[i] =(unsigned long) & ptr[i];
    }

    return;
}

__global__ void
kernel_test1_read(char* _ptr, char* end_ptr, unsigned int* err, unsigned long* err_addr,
		  unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned long* ptr = (unsigned long*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned long*) end_ptr) {
	return;
    }


    for (i = 0;i < BLOCKSIZE/sizeof(unsigned long); i++){
	if (ptr[i] != (unsigned long)& ptr[i]){
	    RECORD_ERR(err, &ptr[i], (unsigned long)&ptr[i], ptr[i]);
	}
    }

    return;
}



void
test1(char* ptr, unsigned int tot_num_blocks)
{


    unsigned int i;
    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;

    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_test1_write<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, err_count); SYNC_CUERR;
	SHOW_PROGRESS("test1 on writing", i, tot_num_blocks);

    }

    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_test1_read<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
	error_checking("test1 on reading",  i);
	SHOW_PROGRESS("test1 on reading", i, tot_num_blocks);

    }


    return;

}


/******************************************************************************
 * Test 2 [Moving inversions, ones&zeros]
 * This test uses the moving inversions algorithm with patterns of all
 * ones and zeros.
 *
 ****************************************************************************/

void
test2(char* ptr, unsigned int tot_num_blocks)
{
    unsigned int p1 = 0;
    unsigned int p2 = ~p1;


    DEBUG_PRINTF("Test2: Moving inversions test, with pattern 0x%x and 0x%x\n", p1, p2);
    move_inv_test(ptr, tot_num_blocks, p1, p2);
    DEBUG_PRINTF("Test2: Moving inversions test, with pattern 0x%x and 0x%x\n", p2, p1);
    move_inv_test(ptr, tot_num_blocks, p2, p1);

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
test3(char* ptr, unsigned int tot_num_blocks)
{
    unsigned int p0=0x80;
    unsigned int p1 = p0 | (p0 << 8) | (p0 << 16) | (p0 << 24);
    unsigned int p2 = ~p1;

    DEBUG_PRINTF("Test3: Moving inversions test, with pattern 0x%x and 0x%x\n", p1, p2);
    move_inv_test(ptr, tot_num_blocks, p1, p2);
    DEBUG_PRINTF("Test3: Moving inversions test, with pattern 0x%x and 0x%x\n", p2, p1);
    move_inv_test(ptr, tot_num_blocks, p2, p1);

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

void
test4(char* ptr, unsigned int tot_num_blocks)
{
    unsigned int p1;
    if (global_pattern == 0){
	p1 = get_random_num();
    }else{
	p1 = global_pattern;
    }

    unsigned int p2 = ~p1;
    unsigned int err = 0;
    unsigned int iteration = 0;

    DEBUG_PRINTF("Test4: Moving inversions test, with random pattern 0x%x and 0x%x\n", p1, p2);

 repeat:
    err += move_inv_test(ptr, tot_num_blocks, p1, p2);

    if (err == 0 && iteration == 0){
	return;
    }
    if (iteration < MAX_ITERATION){
	PRINTF("%dth repeating test4 because there are %d errors found in last run\n", iteration, err);
	iteration++;
	err = 0;
	goto repeat;
    }
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


__global__ void
kernel_test5_init(char* _ptr, char* end_ptr)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    unsigned int p1 = 1;
    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i+=16){
	unsigned int p2 = ~p1;

	ptr[i] = p1;
	ptr[i+1] = p1;
	ptr[i+2] = p2;
	ptr[i+3] = p2;
	ptr[i+4] = p1;
	ptr[i+5] = p1;
	ptr[i+6] = p2;
	ptr[i+7] = p2;
	ptr[i+8] = p1;
	ptr[i+9] = p1;
	ptr[i+10] = p2;
	ptr[i+11] = p2;
	ptr[i+12] = p1;
	ptr[i+13] = p1;
	ptr[i+14] = p2;
	ptr[i+15] = p2;

	p1 = p1<<1;
	if (p1 == 0){
	    p1 = 1;
	}
    }

    return;
}


__global__ void
kernel_test5_move(char* _ptr, char* end_ptr)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    unsigned int half_count = BLOCKSIZE/sizeof(unsigned int)/2;
    unsigned int* ptr_mid = ptr + half_count;

    for (i = 0;i < half_count; i++){
	ptr_mid[i] = ptr[i];
    }

    for (i=0;i < half_count - 8; i++){
	ptr[i + 8] = ptr_mid[i];
    }

    for (i=0;i < 8; i++){
	ptr[i] = ptr_mid[half_count - 8 + i];
    }

    return;
}


__global__ void
kernel_test5_check(char* _ptr, char* end_ptr, unsigned int* err, unsigned long* err_addr,
		   unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    for (i=0;i < BLOCKSIZE/sizeof(unsigned int); i+=2){
	if (ptr[i] != ptr[i+1]){
	    RECORD_ERR(err, &ptr[i], ptr[i+1], ptr[i]);
	}
    }

    return;
}



void
test5(char* ptr, unsigned int tot_num_blocks)
{

    unsigned int i;
    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;

    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_test5_init<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr); SYNC_CUERR;
	SHOW_PROGRESS("test5[init]", i, tot_num_blocks);
    }


    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_test5_move<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr); SYNC_CUERR;
	SHOW_PROGRESS("test5[move]", i, tot_num_blocks);
    }


    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_test5_check<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
	error_checking("test5[check]",  i);
	SHOW_PROGRESS("test5[check]", i, tot_num_blocks);
    }

    return;

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



__global__ void
kernel_movinv32_write(char* _ptr, char* end_ptr, unsigned int pattern,
		unsigned int lb, unsigned int sval, unsigned int offset)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    unsigned int k = offset;
    unsigned pat = pattern;
    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	ptr[i] = pat;
	k++;
	if (k >= 32){
	    k=0;
	    pat = lb;
	}else{
	    pat = pat << 1;
	    pat |= sval;
	}
    }

    return;
}


__global__ void
kernel_movinv32_readwrite(char* _ptr, char* end_ptr, unsigned int pattern,
			  unsigned int lb, unsigned int sval, unsigned int offset, unsigned int * err,
			  unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    unsigned int k = offset;
    unsigned pat = pattern;
    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (ptr[i] != pat){
	    RECORD_ERR(err, &ptr[i], pat, ptr[i]);
	}

	ptr[i] = ~pat;

	k++;
	if (k >= 32){
	    k=0;
	    pat = lb;
	}else{
	    pat = pat << 1;
	    pat |= sval;
	}
    }

    return;
}



__global__ void
kernel_movinv32_read(char* _ptr, char* end_ptr, unsigned int pattern,
		     unsigned int lb, unsigned int sval, unsigned int offset, unsigned int * err,
		     unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    unsigned int k = offset;
    unsigned pat = pattern;
    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (ptr[i] != ~pat){
	    RECORD_ERR(err, &ptr[i], ~pat, ptr[i]);
	}

	k++;
	if (k >= 32){
	    k=0;
	    pat = lb;
	}else{
	    pat = pat << 1;
	    pat |= sval;
	}
    }

    return;
}



void
movinv32(char* ptr, unsigned int tot_num_blocks, unsigned int pattern,
	 unsigned int lb, unsigned int sval, unsigned int offset)
{

    unsigned int i;

    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;

    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_movinv32_write<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, pattern, lb,sval, offset); SYNC_CUERR;
	SHOW_PROGRESS("test6[moving inversion 32 write]", i, tot_num_blocks);
    }

    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_movinv32_readwrite<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, pattern, lb,sval, offset, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
	error_checking("test6[moving inversion 32 readwrite]",  i);
	SHOW_PROGRESS("test6[moving inversion 32 readwrite]", i, tot_num_blocks);
    }

   for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
       dim3 grid;
       grid.x= GRIDSIZE;
       kernel_movinv32_read<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, pattern, lb,sval, offset, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
       error_checking("test6[moving inversion 32 read]",  i);
       SHOW_PROGRESS("test6[moving inversion 32 read]", i, tot_num_blocks);
   }

    return;

}


void
test6(char* ptr, unsigned int tot_num_blocks)
{
    unsigned int i;

    unsigned int pattern;

    for (i= 0, pattern = 1;i < 32; pattern = pattern << 1, i++){

	DEBUG_PRINTF("Test6[move inversion 32 bits test]: pattern =0x%x, offset=%d\n", pattern, i);
	movinv32(ptr, tot_num_blocks, pattern, 1, 0, i);
	DEBUG_PRINTF("Test6[move inversion 32 bits test]: pattern =0x%x, offset=%d\n", ~pattern, i);
	movinv32(ptr, tot_num_blocks, ~pattern, 0xfffffffe, 1, i);

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




__global__ void
kernel_test7_write(char* _ptr, char* end_ptr, char* _start_ptr, unsigned int* err)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
    unsigned int* start_ptr = (unsigned int*) _start_ptr;

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }


    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	ptr[i] = start_ptr[i];
    }

    return;
}



__global__ void
kernel_test7_readwrite(char* _ptr, char* end_ptr, char* _start_ptr, unsigned int* err,
		       unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
    unsigned int* start_ptr = (unsigned int*) _start_ptr;

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }


    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (ptr[i] != start_ptr[i]){
	    RECORD_ERR(err, &ptr[i], start_ptr[i], ptr[i]);
	}
	ptr[i] = ~(start_ptr[i]);
    }

    return;
}

__global__ void
kernel_test7_read(char* _ptr, char* end_ptr, char* _start_ptr, unsigned int* err, unsigned long* err_addr,
		  unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
    unsigned int* start_ptr = (unsigned int*) _start_ptr;

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }


    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (ptr[i] != ~(start_ptr[i])){
	    RECORD_ERR(err, &ptr[i], ~(start_ptr[i]), ptr[i]);
	}
    }

    return;
}


void
test7(char* ptr, unsigned int tot_num_blocks)
{

    unsigned int* host_buf;
    host_buf = (unsigned int*)malloc(BLOCKSIZE);
    unsigned int err = 0;
    unsigned int i;
    unsigned int iteration = 0;

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int);i++){
	host_buf[i] = get_random_num();
    }

    cudaMemcpy(ptr, host_buf, BLOCKSIZE, cudaMemcpyHostToDevice);


    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;

 repeat:

    for (i=1;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_test7_write<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, ptr, err_count); SYNC_CUERR;
	SHOW_PROGRESS("test7_write", i, tot_num_blocks);
    }


    for (i=1;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_test7_readwrite<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, ptr, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
	err += error_checking("test7_readwrite",  i);
	SHOW_PROGRESS("test7_readwrite", i, tot_num_blocks);
    }


    for (i=1;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_test7_read<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, ptr, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
	err += error_checking("test7_read",  i);
	SHOW_PROGRESS("test7_read", i, tot_num_blocks);
    }


    if (err == 0 && iteration == 0){
	return;
    }
    if (iteration < MAX_ITERATION){
	PRINTF("%dth repeating test7 because there are %d errors found in last run\n", iteration, err);
	iteration++;
	err = 0;
	goto repeat;
    }

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



__global__ void
kernel_modtest_write(char* _ptr, char* end_ptr, unsigned int offset, unsigned int p1, unsigned int p2)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    for (i = offset;i < BLOCKSIZE/sizeof(unsigned int); i+=MOD_SZ){
	ptr[i] =p1;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (i % MOD_SZ != offset){
	    ptr[i] =p2;
	}
    }

    return;
}


__global__ void
kernel_modtest_read(char* _ptr, char* end_ptr, unsigned int offset, unsigned int p1, unsigned int* err,
		    unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    for (i = offset;i < BLOCKSIZE/sizeof(unsigned int); i+=MOD_SZ){
	if (ptr[i] !=p1){
	    RECORD_ERR(err, &ptr[i], p1, ptr[i]);
	}
    }

    return;
}

unsigned int
modtest(char* ptr, unsigned int tot_num_blocks, unsigned int offset, unsigned int p1, unsigned int p2)
{

    unsigned int i;
    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;
    unsigned int err = 0;

    for (i= 0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_modtest_write<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, offset, p1, p2); SYNC_CUERR;
	SHOW_PROGRESS("test8[mod test, write]", i, tot_num_blocks);
    }

    for (i= 0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_modtest_read<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, offset, p1, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
	err += error_checking("test8[mod test, read", i);
	SHOW_PROGRESS("test8[mod test, read]", i, tot_num_blocks);
    }

    return err;

}

void
test8(char* ptr, unsigned int tot_num_blocks)
{
    unsigned int i;
    unsigned int err = 0;
    unsigned int iteration = 0;

    unsigned int p1;
    if (global_pattern){
	p1 = global_pattern;
    }else{
	p1= get_random_num();
    }
    unsigned int p2 = ~p1;

 repeat:

    PRINTF("test8[mod test]: p1=0x%x, p2=0x%x\n", p1,p2);
    for (i = 0;i < MOD_SZ; i++){
	err += modtest(ptr, tot_num_blocks,i, p1, p2);
    }

    if (err == 0 && iteration == 0){
	return;
    }

    if (iteration < MAX_ITERATION){
	PRINTF("%dth repeating test8 because there are %d errors found in last run, p1=%x, p2=%x\n", iteration, err, p1, p2);
	iteration++;
	err = 0;
	goto repeat;
    }
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
test9(char* ptr, unsigned int tot_num_blocks)
{

    unsigned int p1 = 0;
    unsigned int p2 = ~p1;

    unsigned int i;
    char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;

    for (i= 0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_move_inv_write<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, p1); SYNC_CUERR;
	SHOW_PROGRESS("test9[bit fade test, write]", i, tot_num_blocks);
    }

    DEBUG_PRINTF("sleeping for 90 minutes\n");
    sleep(60*90);

    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_move_inv_readwrite<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, p1, p2, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
	error_checking("test9[bit fade test, readwrite]",  i);
	SHOW_PROGRESS("test9[bit fade test, readwrite]", i, tot_num_blocks);
    }

    DEBUG_PRINTF("sleeping for 90 minutes\n");
    sleep(60*90);
    for (i=0;i < tot_num_blocks; i+= GRIDSIZE){
	dim3 grid;
	grid.x= GRIDSIZE;
	kernel_move_inv_read<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, p2, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
	error_checking("test9[bit fade test, read]",  i);
	SHOW_PROGRESS("test9[bit fade test, read]", i, tot_num_blocks);
    }

    return;
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

#define TYPE unsigned long
__global__ void test10_kernel_write(char* ptr, int memsize, TYPE p1)
{
    int i;
    int avenumber = memsize/(gridDim.x*gridDim.y);
    TYPE* mybuf = (TYPE*)(ptr + blockIdx.x* avenumber);
    int n = avenumber/(blockDim.x*sizeof(TYPE));

    for(i=0;i < n;i++){
        int index = i*blockDim.x + threadIdx.x;
        mybuf[index]= p1;
    }
    int index = n*blockDim.x + threadIdx.x;
    if (index*sizeof(TYPE) < avenumber){
        mybuf[index] = p1;
    }

    return;
}

__global__ void test10_kernel_readwrite(char* ptr, int memsize, TYPE p1, TYPE p2,  unsigned int* err,
					unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    int i;
    int avenumber = memsize/(gridDim.x*gridDim.y);
    TYPE* mybuf = (TYPE*)(ptr + blockIdx.x* avenumber);
    int n = avenumber/(blockDim.x*sizeof(TYPE));
    TYPE localp;

    for(i=0;i < n;i++){
        int index = i*blockDim.x + threadIdx.x;
        localp = mybuf[index];
        if (localp != p1){
	    RECORD_ERR(err, &mybuf[index], p1, localp);
	}
	mybuf[index] = p2;
    }
    int index = n*blockDim.x + threadIdx.x;
    if (index*sizeof(TYPE) < avenumber){
	localp = mybuf[index];
	if (localp!= p1){
	    RECORD_ERR(err, &mybuf[index], p1, localp);
	}
	mybuf[index] = p2;
    }

    return;
}


////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

#define STRESS_BLOCKSIZE 64
#define STRESS_GRIDSIZE (1024*32)

void test10(char* ptr, unsigned int tot_num_blocks)
{
    TYPE p1;
    if (global_pattern_long){
	p1 = global_pattern_long;
    }else{
	p1 = get_random_num_long();
    }
    TYPE p2 = ~p1;
    cudaStream_t stream;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    int n = num_iterations;
    float elapsedtime;
    dim3 gridDim(STRESS_GRIDSIZE);
    dim3 blockDim(STRESS_BLOCKSIZE);
    checkCudaErrors(cudaEventRecord(start, stream));

    PRINTF("Test10 with pattern=0x%lx\n", p1);
    test10_kernel_write<<<gridDim, blockDim, 0, stream>>>(ptr, tot_num_blocks*BLOCKSIZE, p1); SYNC_CUERR;
    for(int i =0;i < n ;i ++){
	test10_kernel_readwrite<<<gridDim, blockDim, 0, stream>>>(ptr, tot_num_blocks*BLOCKSIZE, p1, p2,
								  err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
	p1 = ~p1;
	p2 = ~p2;

    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    error_checking("test10[Memory stress test]",  0);
    cudaEventElapsedTime(&elapsedtime, start, stop);
    DEBUG_PRINTF("test10: elapsedtime=%f, bandwidth=%f GB/s\n", elapsedtime, (2*n+1)*tot_num_blocks/elapsedtime);

   cudaEventDestroy(start);
   cudaEventDestroy(stop);

   cudaStreamDestroy(stream);

#if 0
    TYPE* host_buf = (TYPE*)malloc(tot_num_blocks*BLOCKSIZE);
    if (host_buf == NULL){
	printf("ERROR: malloc failed for host_buf\n");
	exit(ERR_GENERAL);
    }
    memset(host_buf, 0, tot_num_blocks* BLOCKSIZE);
    cudaMemcpy(host_buf, ptr, tot_num_blocks*BLOCKSIZE, cudaMemcpyDeviceToHost);
    for(unsigned long i=0;i < (tot_num_blocks*BLOCKSIZE)/sizeof(TYPE) ;i ++){
	if (host_buf[i] != p1){
	    PRINTF("ERROR: data not match for i=%d, expecting 0x%x, current value=0x%x\n", i, p1, host_buf[i]);
	    free(host_buf);
	    exit(ERR_GENERAL);
	}
    }
    printf("all data match\n");
    free(host_buf);
#endif


}


cuda_memtest_t cuda_memtests[]={
    {test0, (char*)"Test0 [Walking 1 bit]",			1},
    {test1, (char*)"Test1 [Own address test]",			1},
    {test2, (char*)"Test2 [Moving inversions, ones&zeros]",	1},
    {test3, (char*)"Test3 [Moving inversions, 8 bit pat]",	1},
    {test4, (char*)"Test4 [Moving inversions, random pattern]",1},
    {test5, (char*)"Test5 [Block move, 64 moves]",		1},
    {test6, (char*)"Test6 [Moving inversions, 32 bit pat]",	1},
    {test7, (char*)"Test7 [Random number sequence]",		1},
    {test8, (char*)"Test8 [Modulo 20, random pattern]",	1},
    {test9, (char*)"Test9 [Bit fade test]",			0},
    {test10, (char*)"Test10 [Memory stress test]",		1},

};



void
allocate_small_mem(void)
{
    cudaMalloc((void**)&err_count, sizeof(unsigned int)); CUERR;
    cudaMemset(err_count, 0, sizeof(unsigned int)); CUERR;

    cudaMalloc((void**)&err_addr, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT);CUERR;
    cudaMemset(err_addr, 0, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT);

    cudaMalloc((void**)&err_expect, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT);CUERR;
    cudaMemset(err_expect, 0, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT);

    cudaMalloc((void**)&err_current, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT);CUERR;
    cudaMemset(err_current, 0, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT);

    cudaMalloc((void**)&err_second_read, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT);CUERR;
    cudaMemset(err_second_read, 0, sizeof(unsigned long)*MAX_ERR_RECORD_COUNT);
}

void
run_tests(char* ptr, unsigned int tot_num_blocks)
{

    struct timeval  t0, t1;
    unsigned int i;

    unsigned int pass = 0;

    while(1){

	for (i = 0;i < DIM(cuda_memtests); i++){
	    if (cuda_memtests[i].enabled){
		PRINTF("%s\n", cuda_memtests[i].desc);
		gettimeofday(&t0, NULL);
		cuda_memtests[i].func(ptr, tot_num_blocks);
		gettimeofday(&t1, NULL);
		PRINTF("Test%d finished in %.1f seconds\n", i, TDIFF(t1, t0));
	    }//if
	}//for

	if (num_passes <= 0){
	    continue;
	}

	pass ++;
	if (pass >= num_passes){
	    break;
	}
    }//while

}
