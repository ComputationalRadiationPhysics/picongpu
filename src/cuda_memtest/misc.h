#pragma once

#include <stdlib.h>
#include <assert.h>
#include "cuda_memtest.h"

void update_temperature(void);

void get_serial_number(unsigned int devIdx, char* serial);

#define NV_DRIVER_VER_FILE "/proc/driver/nvidia/version"
void get_driver_info(char* info, unsigned int len);
