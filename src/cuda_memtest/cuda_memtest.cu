/*
 * Illinois Open Source License
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright � 2009,    University of Illinois.  All rights reserved.
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

#include "cuda_memtest.h"
#include <cublas.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <signal.h>
#include <cuda.h>

#define MAX_NUM_GPUS 8
bool useMappedMemory;
void* mappedHostPtr;
char hostname[64];
unsigned int verbose =0;
unsigned int interactive =0;
extern cuda_memtest_t cuda_memtests[11];
unsigned int max_num_blocks = 0;
unsigned int exit_on_error = 0;
unsigned int monitor_temp = 0;
unsigned int monitor_interval = 5;
unsigned int email_notification = 0;
unsigned int global_pattern = 0;
unsigned long global_pattern_long = 0;
char emails[128];
unsigned int report_interval = 1800;  //senconds
unsigned long long serial_number = 0;
unsigned int num_iterations = 1000;
unsigned int num_passes = 0;
unsigned int healthy_threads = 0;
unsigned int disable_serial_number = 0;
__thread unsigned int gpu_idx;
char driver_info[MAX_STR_LEN];

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t atomic_mutex = PTHREAD_MUTEX_INITIALIZER;

void run_tests(char*, unsigned int);
extern void update_temperature(void);
extern unsigned long long get_serial_number(void);
extern void allocate_small_mem(void);

typedef struct arg_s{
    unsigned int device;
}arg_t;

/*

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    size_t totalConstMem;
    int major;
    int minor;
    int clockRate;
    size_t textureAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
}
*/
void
display_device_info(struct cudaDeviceProp* prop)
{
    PRINTF("Device name=%s, global memory size=%llu\n", prop->name, prop->totalGlobalMem);
    return;
}


void
atomic_inc(unsigned int* value)
{
    pthread_mutex_lock(&atomic_mutex);
    (*value)= (*value) + 1;
    pthread_mutex_unlock(&atomic_mutex);
}

unsigned int
atomic_read(unsigned int* value)
{
    unsigned int ret;

    pthread_mutex_lock(&atomic_mutex);
    ret = *value;
    pthread_mutex_unlock(&atomic_mutex);

    return ret;
}

void*
thread_func(void* _arg)
{

    arg_t* arg = (arg_t*)_arg;
    unsigned int device = arg->device;
    gpu_idx = device;



    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device); CUERR;

    display_device_info(&prop);

    unsigned long totmem = prop.totalGlobalMem;

    PRINTF("major=%d, minor=%d\n", prop.major, prop.minor);

    //need to leave a little headroom or later calls will fail
    unsigned int tot_num_blocks = totmem/BLOCKSIZE -16;
    if (max_num_blocks != 0){
	tot_num_blocks = MIN(max_num_blocks+16, tot_num_blocks);
    }


    cudaSetDevice(device);
    cudaThreadSynchronize();
    CUERR;

    PRINTF("Attached to device %d successfully.\n", device);

    size_t free, total;
    cudaMemGetInfo(&free, &total);

    allocate_small_mem();

    char* ptr = NULL;

    tot_num_blocks = MIN(tot_num_blocks, free/BLOCKSIZE - 16);
    do{
        tot_num_blocks -= 16 ; //magic number 16 MB
        DEBUG_PRINTF("Trying to allocate %d MB\n", tot_num_blocks);
        if (tot_num_blocks <= 0){
            FPRINTF("ERROR: cannot allocate any memory from GPU\n");
            exit(ERR_GENERAL);
        }
        if(useMappedMemory)
        {
            //create cuda mapped memory
            cudaHostAlloc((void**)&mappedHostPtr,tot_num_blocks* BLOCKSIZE,cudaHostAllocMapped);
            cudaHostGetDevicePointer(&ptr,mappedHostPtr,0);
        }
        else
        {
            cudaMalloc((void**)&ptr, tot_num_blocks* BLOCKSIZE);
        }
    }while(cudaGetLastError() != cudaSuccess);

    PRINTF("Allocated %d MB\n", tot_num_blocks);

    atomic_inc(&healthy_threads);
    run_tests(ptr, tot_num_blocks);

    return NULL;

}


void*
temp_monitor_thread_func(void* arg)
{
    do{
	update_temperature();
	sleep(monitor_interval);
    }while(1);

}


void list_tests_info(void)
{
    int i;
    for (i = 0;i < DIM(cuda_memtests); i++){
	printf("%s %s\n", cuda_memtests[i].desc, cuda_memtests[i].enabled?"":" ==disabled by default==");
    }
    return;
}


void
usage(char** argv)
{

    char example_usage[] =
	"run on default setting:       ./cuda_memtest\n"
	"run on stress test only:      ./cuda_memtest --stress\n";

    printf("Usage:%s [options]\n", argv[0]);
    printf("options:\n");
    printf("--mappedMem                 run all checks with cuda mapped memory instead of native device memory\n");
    printf("--silent                    Do not print out progress message (default)\n");
    printf("--device <idx>              Designate one device for test\n");
    printf("--interactive               Progress info will be printed in the same line\n");
    printf("--disable_all               Disable all tests\n");
    printf("--enable_test <test_idx>    Enable the test <test_idx>\n");
    printf("--disable_test <test_idx>   Disable the test <test_idx>\n");
    printf("--max_num_blocks <n>        Set the maximum of blocks of memory to test\n");
    printf("                            1 block = 1 MB in here\n");
    printf("--exit_on_error             When finding error, print error message and exit\n");
    printf("--monitor_temp <interval>   Monitoring temperature, the temperature will be updated every <interval> seconds\n");
    printf("                            This feature is experimental\n");
    printf("--emails <a@b,c@d,...>      Setting email notification\n");
    printf("--report_interval <n>       Setting the interval in seconds between email notifications(default 1800)\n");
    printf("--pattern <pattern>         Manually set test pattern for test4/test8/test10\n");
    printf("--list_tests                List all test descriptions\n");
    printf("--num_iterations <n>        Set the number of iterations (only effective on test0 and test10)\n");
    printf("--num_passes <n>            Set the number of test passes (this affects all tests)\n");
    printf("--disable_serial_number     Disable reporting serial number\n");
    printf("--verbose <n>               Setting verbose level\n");
    printf("                              0 -- Print out test start and end message only (default)\n");
    printf("                              1 -- Print out pattern messages in test\n");
    printf("                              2 -- Print out progress messages\n");
    printf("--stress                    Stress test. Equivalent to --disable_all --enable_test 10 --exit_on_error\n");
    printf("--help                      Print this message\n");
    printf("\nExample usage:\n\n");
    printf("%s\n", example_usage);

    exit(ERR_GENERAL);
}


int
main(int argc, char** argv)
{
    int i;
    useMappedMemory=false;
    mappedHostPtr=NULL;

    if (argc >=2 ){
	if( strcmp(argv[1], "--help")== 0){
	    usage(argv);
	}
    }

    if(gethostname(hostname, 64) !=0){
	fprintf(stderr, "ERROR: gethostname() returns error\n");
	exit(ERR_GENERAL);
    }

    for(i=0;i < 64; i++){
	if (hostname[i] == '.'){
	    hostname[i] = 0;
	    break;
	}
    }

    PRINTF("Running cuda memtest, version %s\n", VERSION);
    int device = -1;
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);CUERR;

    if (num_gpus == 0){
	fprintf(stderr,"ERROR: no GPUs found\n");
	exit(ERR_GENERAL);
    }


    for (i =1;i < argc; i++){

	if( strcmp(argv[i], "--help")== 0){
	    usage(argv);
	}

    if( strcmp(argv[i], "--mappedMem")== 0){
	    useMappedMemory=true;
        continue;
	}

	if( strcmp(argv[i], "--verbose") == 0){
	    if (i+1 >= argc){
		usage(argv);
	    }
	    verbose = atoi(argv[i+1]);
	    i++;
	    continue;

	}
	if (strcmp(argv[i], "--silent") == 0){
	    verbose = 0;
	    continue;
	}
	if (strcmp(argv[i], "--interactive") == 0){
	    interactive = 1;
	    continue;
	}
	if (strcmp(argv[i], "--noninteractive") == 0){
	    interactive = 0;
	    continue;
	}
	if (strcmp(argv[i], "--enable_test") == 0){
	    if (i+1 >= argc){
		usage(argv);
	    }
	    int idx = atoi(argv[i+1]);
	    if (idx >= DIM(cuda_memtests)){
		fprintf(stderr, "Error: invalid test id\n");
		usage(argv);
	    }

	    cuda_memtests[idx].enabled = 1;

	    i++;
	    continue;
	}
	if (strcmp(argv[i], "--disable_test") == 0){
	    if (i+1 >= argc){
		usage(argv);
	    }
	    int idx = atoi(argv[i+1]);
	    if (idx >= DIM(cuda_memtests)){
		fprintf(stderr, "Error: invalid test id\n");
		usage(argv);
	    }

	    cuda_memtests[idx].enabled = 0;
	    i++;
	    continue;
	}
	if (strcmp(argv[i], "--disable_all") == 0){
	    int k;
	    for (k=0;k < DIM(cuda_memtests);k++){
		cuda_memtests[k].enabled = 0;
	    }
	    continue;
	}

	if (strcmp(argv[i], "--device") == 0){
	    if (i+1 >= argc){
		usage(argv);
	    }
	    device = atoi(argv[i+1]);
	    i++;
	    num_gpus = 1;
	    continue;
	}

	if (strcmp(argv[i], "--max_num_blocks") == 0){
	    if (i+1 >= argc){
		usage(argv);
	    }
	    max_num_blocks = atoi(argv[i+1]);
	    i++;
	    continue;
	}

	if (strcmp(argv[i], "--exit_on_error") == 0){
	    exit_on_error = 1;
	    continue;
	}

	if (strcmp(argv[i], "--monitor_temp") == 0){
	    monitor_temp =1;
	    if (i+1 >= argc){
		usage(argv);
	    }
	    monitor_interval = atoi(argv[i+1]);
	    i++;
	    continue;
	}
	if (strcmp(argv[i], "--pattern") == 0){
	    if (i+1 >= argc){
		usage(argv);
	    }
	    sscanf(argv[i+1], "0x%lx", &global_pattern_long);
	    if (global_pattern_long ==0){
		printf("ERROR: global test pattern cannot be zero\n");
		usage(argv);
	    }
	    printf("Using global test pattern: 0x%lx\n", global_pattern_long);
	    global_pattern = (unsigned long)global_pattern_long;
	    i++;
	    continue;
	}
	if (strcmp(argv[i], "--emails") == 0){
	    email_notification =1;

	    struct stat statbuf;
	    if (stat(MAILFILE, &statbuf)!=0){
		fprintf(stderr, "ERROR: stating mail unitility(%s) failed\n", MAILFILE);
		usage(argv);
	    }

	    if( !(S_IXOTH & statbuf.st_mode)){
		fprintf(stderr, "ERROR: no permission on exeution on the mail utility\n");
		usage(argv);
	    }


	    if (i+1 >= argc){
		usage(argv);
	    }
	    if ( strlen( argv[i+1]) > sizeof(emails)){
		fprintf(stderr, "ERROR: email string too long\n");
		usage(argv);
	    }
	    strcpy(emails, argv[i+1]);
	    i++;
	    continue;
	}
	if (strcmp(argv[i], "--report_interval") == 0){

	    if (i+1 >= argc){
		usage(argv);
	    }
	    report_interval = atoi(argv[i+1]);
	    i++;
	    continue;
	}

	if (strcmp(argv[i], "--num_iterations") == 0){

	    if (i+1 >= argc){
		usage(argv);
	    }
	    num_iterations = atoi(argv[i+1]);
	    if (num_iterations <= 0){
		printf("ERROR: invalid number of iterations\n");
		usage(argv);
	    }
	    i++;
	    continue;
	}

	if (strcmp(argv[i], "--num_passes") == 0){

	    if (i+1 >= argc){
		usage(argv);
	    }
	    num_passes = atoi(argv[i+1]);
	    if (num_passes <= 0){
		printf("ERROR: invalid number of passes\n");
		usage(argv);
	    }
	    i++;
	    continue;
	}

	if (strcmp(argv[i], "--disable_serial_number") == 0){
	    disable_serial_number= 1;
	    continue;
	}

	if (strcmp(argv[i], "--stress") == 0){
	    //equal to "--disable_all --enable_test 10 --exit_on_error"
	    int k;
	    for (k=0;k < DIM(cuda_memtests);k++){
		cuda_memtests[k].enabled = 0;
	    }
	    cuda_memtests[10].enabled = 1;
	    exit_on_error = 1;
	    continue;
	}

	if (strcmp(argv[i], "--list_tests") == 0){
	    list_tests_info();
	    return 0;
	}
	fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
	usage(argv);
    }

    if (!disable_serial_number){
	serial_number  = get_serial_number();
    }

    get_driver_info(driver_info, MAX_STR_LEN);

    PRINTF("num_gpus=%d\n", num_gpus);
    if(num_gpus > MAX_NUM_GPUS){
	fprintf(stderr, "Error: max number of GPUs (%d) exceeded: %d\n", MAX_NUM_GPUS, num_gpus);
    }
    pthread_t temp_pid;
    if (monitor_temp){
	if (pthread_create(&temp_pid, NULL, temp_monitor_thread_func, NULL)  != 0){
	    printf("ERROR: creating thread for temperature monitoring failed\n");
	    exit(ERR_GENERAL);
	}
    }

    arg_t args[MAX_NUM_GPUS];
    pthread_t pid[MAX_NUM_GPUS];

    if (device != -1){ //device set, only 1 GPU
	args[0].device = device;
	pthread_create(&pid[0], NULL, thread_func, (void*)&args[0]);
    }else{//multiple GPUs
	for (i=0;i < num_gpus;i++){
	    args[i].device = i;
	    pthread_create(&pid[i], NULL, thread_func, (void*)&args[i]);
	}

    }

    struct timeval t0, t1;
    int ht=0;
    double wait_time = 500;
    gettimeofday(&t0, NULL);

    while(1){
	ht = atomic_read(&healthy_threads);
	if (ht == num_gpus){
	    break;
	}

	gettimeofday(&t1, NULL);
	double passed_time = TDIFF(t1, t0);
	if (passed_time >= wait_time){
	    break;
	}
	sleep(1);
    }

    if (ht < num_gpus){
	printf("ERROR: Some GPU threads are not progressing (healthy_threads=%d, num_gpus=%d)\n", ht, num_gpus);
	fflush(stdout); fflush(stderr);
	for(i=0;i < num_gpus;i++){
		pthread_kill(pid[i], SIGTERM);
	}
	exit(ERR_BAD_STATE);
    }


    for(i=0;i < num_gpus;i++){
	pthread_join(pid[i], NULL);
    }

    printf("main thread: Program exits\n");

    return 0;
}

