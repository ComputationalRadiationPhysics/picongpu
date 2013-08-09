
#ifndef __OCL_TESTS_H__
#define __OCL_TESTS_H__


#ifdef __cplusplus
extern "C" {
#endif
  
#define TYPE unsigned long
#define MAX_ERR_RECORD_COUNT 10
#define MAX_NUM_DEVICES 16
#define N 128
#define MB (1024*1024)
#define MAX_NUM_KERNELS  16
#define MOD_SZ 20
#define MAX_ERR_ITERATION 3
#define BLOCKSIZE ((unsigned long)(1024*1024))

  char* print_cl_errstring(cl_int err);
  
#define TDIFF(tb, ta) (tb.tv_sec - ta.tv_sec + 0.000001*(tb.tv_usec - ta.tv_usec))

#define RECORD_ERR(err, p, expect, current) do{         \
    unsigned int idx = atomicAdd(err, 1);		\
    idx = idx % MAX_ERR_RECORD_COUNT;			\
    err_addr[idx] = (unsigned long)p;			\
    err_expect[idx] = (unsigned long)expect;		\
    err_current[idx] = (unsigned long)current;		\
    err_second_read[idx] = (unsigned long)(*p);		\
  }while(0) 
  
#define PRINTF(fmt,...) do{                                             \
    pthread_mutex_lock(&mutex);						\
    printf("[%s][%s][%d]:"fmt, time_string(), hostname, mc->device_idx, ##__VA_ARGS__); \
    fflush(stdout);							\
    pthread_mutex_unlock(&mutex);                                       \
  } while(0) 


  
#define ERR_BAD_STATE  -1
#define ERR_GENERAL -999
  
#define CLERR if (rc != CL_SUCCESS){					\
    printf("ERROR: opencl call failed with rc(%d), line %d, file %s\n", rc, __LINE__, __FILE__); \
    printf("Error: %s\n", print_cl_errstring(rc));			\
    exit(1);                                                            \
  }  

#define DIM(x) (sizeof(x)/sizeof(x[0]))
  
  typedef struct memtest_control_s{
    cl_context context;
    cl_uint device_idx;
    cl_device_id device;
    cl_command_queue queue;
    cl_mem device_mem;
    cl_mem err_count;
    cl_mem err_addr;
    cl_mem err_expect;
    cl_mem err_current;
    cl_mem err_second_read;
    cl_ulong mem_size;
    cl_program program;
    cl_event events[MAX_NUM_KERNELS];    
  }memtest_control_t;
  
  void test10(memtest_control_t*);
  void* run_tests(void* arg);

  
  typedef  void (*test_func_t)(memtest_control_t*);
  
  typedef struct cuda_memtest_s{
    test_func_t func;
    const char* desc;
    unsigned int enabled;
  }cuda_memtest_t;


  
#ifdef __cplusplus
}
#endif


#endif

