#include "misc.h"

volatile int gpu_temp[MAX_GPU_NUM];

void update_temperature(void)
{
#if (ENABLE_NVML==1)
    unsigned int deviceCount;
    NVML_CHECK(nvmlDeviceGetCount( &deviceCount ));

    for( unsigned int devIdx = 0; devIdx < deviceCount; ++devIdx )
    {
        nvmlDevice_t devHandle;
        NVML_CHECK(nvmlDeviceGetHandleByIndex( devIdx, &devHandle ));

        unsigned int devTemperature;
        NVML_CHECK(nvmlDeviceGetTemperature( devHandle, NVML_TEMPERATURE_GPU, &devTemperature ));
        gpu_temp[devIdx] = devTemperature;

        DEBUG_PRINTF("temperature updated: (gpu %d) %d \n", devIdx, devTemperature);
    }
#endif
}


void get_serial_number(unsigned int devIdx, char* serial)
{
#if (ENABLE_NVML==1)
    try
    {
        nvmlDevice_t devHandle;
        NVML_CHECK(nvmlDeviceGetHandleByIndex( devIdx, &devHandle ));

        unsigned int serialLength = NVML_DEVICE_SERIAL_BUFFER_SIZE;
        NVML_CHECK(nvmlDeviceGetSerial( devHandle, serial, serialLength ));
    }
    catch(const std::runtime_error& e)
    {
        std::strncpy(
            serial,
            "unknown (NVML runtime error)",
            NVML_DEVICE_SERIAL_BUFFER_SIZE);
        serial[NVML_DEVICE_SERIAL_BUFFER_SIZE-1] = '\0';
    }
#else
    (void)(devIdx);
    (void)(serial);
#endif
}


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
	fclose(file);
	return;
    }
    fclose(file);

    PRINTF("%s", info);

    return;
}
