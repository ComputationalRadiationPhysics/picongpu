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

#ifndef __MEMTEST_H__
#define __MEMTEST_H_

#include <stdio.h>
#include <pthread.h>

#define VERSION "1.2.2"

#define ERR_BAD_STATE  -1
#define ERR_GENERAL -999

extern FILE* stdin;
extern FILE* stdout;
extern FILE* stderr;
extern unsigned int monitor_temp;

extern char hostname[];
extern __thread unsigned int gpu_idx;
extern unsigned int verbose;
extern unsigned int interactive;
extern char* time_string(void);
extern volatile int gpu_temp[];
extern void update_temperature(void);
extern pthread_mutex_t mutex;
extern void get_driver_info(char* info, unsigned int len);

#define MAX_GPU_NUM  4
#define BLOCKSIZE ((unsigned long)(1024*1024))
#define GRIDSIZE 128

#define PRINTF(fmt,...) do{						\
	if (monitor_temp){						\
	    pthread_mutex_lock(&mutex);					\
	    printf("[%s][%s][%d][%d C]:"fmt, time_string(), hostname, gpu_idx, gpu_temp[gpu_idx],##__VA_ARGS__); \
	    pthread_mutex_unlock(&mutex);				\
	}								\
	else{								\
	    pthread_mutex_lock(&mutex);					\
	    printf("[%s][%s][%d]:"fmt, time_string(), hostname, gpu_idx, ##__VA_ARGS__); \
	    pthread_mutex_unlock(&mutex);					\
	}								\
	fflush(stdout);							\
    } while(0) 


#define FPRINTF(fmt,...) do{						\
	if (monitor_temp){					\
	    fprintf(stderr, "[%s][%s][%d][%d C]:"fmt, time_string(), hostname, gpu_idx, gpu_temp[gpu_idx],##__VA_ARGS__); \
	}								\
	else{								\
	    fprintf(stderr, "[%s][%s][%d]:"fmt, time_string(), hostname, gpu_idx, ##__VA_ARGS__); \
	}								\
	fflush(stderr);							\
    } while(0) 



#define DEBUG_PRINTF(fmt,...) do {					\
	if (verbose){							\
	    PRINTF(fmt, ##__VA_ARGS__);					\
	}								\
    }while(0)



#define SHOW_PROGRESS(msg, i, tot_num_blocks)				\
    cudaThreadSynchronize();						\
    unsigned int num_checked_blocks =  i+GRIDSIZE <= tot_num_blocks? i+GRIDSIZE: tot_num_blocks; \
    if (verbose >=2){							\
	if(interactive){ \
	    printf("\r%s[%d]:%s: %d out of %d blocks finished", hostname, gpu_idx, msg, num_checked_blocks, tot_num_blocks ); \
	}else{								\
	    PRINTF("%s: %d out of %d blocks finished\n", msg, num_checked_blocks, tot_num_blocks );  \
	} \
   }		\
    fflush(stdout);					


#define CUERR  do{ cudaError_t cuda_err; \
	if ((cuda_err = cudaGetLastError()) != cudaSuccess) {		\
	    FPRINTF("ERROR: CUDA error: %s, line %d, file %s\n", cudaGetErrorString(cuda_err),  __LINE__, __FILE__); \
	    PRINTF("ERROR: CUDA error: %s, line %d, file %s\n", cudaGetErrorString(cuda_err),  __LINE__, __FILE__); \
	    exit(cuda_err);}}while(0) 

#define SYNC_CUERR  do{ cudaError_t cuda_err; \
	cudaThreadSynchronize(); \
        if ((cuda_err = cudaGetLastError()) != cudaSuccess) {                \
            FPRINTF("ERROR: CUDA error: %s, line %d, file %s\n", cudaGetErrorString(cuda_err),  __LINE__, __FILE__); \
            PRINTF("ERROR: CUDA error: %s, line %d, file %s\n", cudaGetErrorString(cuda_err),  __LINE__, __FILE__); \
            exit(cuda_err);}}while(0) 

#define TDIFF(tb, ta) (tb.tv_sec - ta.tv_sec + 0.000001*(tb.tv_usec - ta.tv_usec))
#define DIM(x) (sizeof(x)/sizeof(x[0]))
#define MIN(x,y) (x < y? x: y)
#define MOD_SZ 20
#define MAILFILE "/bin/mail"
#define MAX_STR_LEN 256

typedef  void (*test_func_t)(char* , unsigned int );

typedef struct cuda_memtest_s{
    test_func_t func;
    char* desc;
    unsigned int enabled;    
}cuda_memtest_t;



#endif
