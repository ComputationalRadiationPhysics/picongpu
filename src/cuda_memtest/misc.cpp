
#include <stdlib.h>
#include <assert.h>
#include "cuda_memtest.h"

volatile int intake_temp[MAX_GPU_NUM/2];
volatile int gpu_temp[MAX_GPU_NUM];

void update_temperature(void)
{
    FILE* file= popen("nvidia-smi|/bin/grep Temperature|gawk '{print $3 $4}'", "r");
    if (file == NULL){
	printf("ERROR: opening pipe failed\n");
	exit(ERR_GENERAL);
    }
    unsigned int i = 0;    
    while(1){
	char buf[256];
	char* s;
	int t;
	int gpu_0,  gpu_1;
	
	s = fgets(buf, 256, file);
	if (s == NULL){
	    break;
	}
	sscanf(buf, ":%d", &t);
	
	s = fgets(buf, 256, file);
	assert(s);
	sscanf(buf, "%dC", &gpu_0);
	
	s = fgets(buf, 256, file);
	assert(s);
	sscanf(buf, "%dC", &gpu_1);
	
	intake_temp[i] = t;
	gpu_temp[2*i]  = gpu_0;
	gpu_temp[2*i + 1]  = gpu_1;	
	i++;
    }
    
    DEBUG_PRINTF("temperature updated: %d %d %d %d \n", 
		 gpu_temp[0], gpu_temp[1], gpu_temp[2], gpu_temp[3]);
	
   pclose(file);
    
}


unsigned long long
get_serial_number(void)
{   
    FILE* file= popen("nvidia-smi|/bin/grep \"Serial Number\"|gawk '{print $4}'", "r");
    if (file == NULL){
	PRINTF("Warning: opening pipe failed for getting serial nubmer\n");
	return 0;
    }
    
    char buf[256];
    char* s;
    int t;
    unsigned long long sn;
    
    s = fgets(buf, 256, file);
    if (s == NULL){
	PRINTF("Warning: Getting serial number failed\n");
	pclose(file);
	return 0;
    }
    sscanf(buf, "%llu", &sn);
    
    PRINTF("Unit serial number: %llu\n", sn);
    
    pclose(file);
    
    return sn;
}


#define NV_DRIVER_VER_FILE "/proc/driver/nvidia/version"
void
get_driver_info(char* info, unsigned int len)
{
    
    FILE* file = fopen(NV_DRIVER_VER_FILE, "r");
    if (file == NULL){
	PRINTF("Warning: Opening %s failed\n", NV_DRIVER_VER_FILE);
	info[0] = 0;
	return;
    }

    if ( fgets(info, len, file) == NULL){
	PRINTF("Warning: reading file failed\n");
	info[0] = 0;
	return;
    }
    
    PRINTF("%s", info);
    
    return;
}
