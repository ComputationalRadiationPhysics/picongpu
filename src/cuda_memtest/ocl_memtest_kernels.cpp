
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define TYPE unsigned long
#define MAX_ERR_RECORD_COUNT 10
#define MOD_SZ 20
#define BLOCKSIZE (1024*1024)

#define RECORD_ERR(err, p, expect, current) do{         \
    unsigned int idx = atom_add(err, 1);		\
    idx = idx % MAX_ERR_RECORD_COUNT;			\
    err_addr[idx] = (unsigned long)p;			\
    err_expect[idx] = (unsigned long)expect;		\
    err_current[idx] = (unsigned long)current;		\
    err_second_read[idx] = (unsigned long)(*p);		\
  }while(0) 

__kernel void
kernel_modtest_write(__global char* ptr, unsigned long memsize, 
		     unsigned int offset, TYPE p1, TYPE p2)
{
  int i;  
  __global TYPE* buf = (__global TYPE*)ptr;
  int idx = get_global_id(0);
  unsigned long n = memsize/sizeof(TYPE);
  int total_num_threads = get_global_size(0);  
  
  for(i=idx;i < n; i+= total_num_threads){
    if ( (i+MOD_SZ-offset)%MOD_SZ == 0){
      buf[i] = p1;
    }else{
      buf[i] = p2;
    }
  }
  
  return;
    
}

__kernel void
kernel_modtest_read(__global char* ptr, unsigned long memsize, 
		    unsigned int offset,  TYPE p1, TYPE p2, 
		    volatile __global unsigned int* err_count,
		    __global unsigned long* err_addr, 
		    __global unsigned long* err_expect,
		    __global unsigned long* err_current, 
		    __global unsigned long* err_second_read)
{
  int i;  
  __global TYPE* buf = (__global TYPE*)ptr;
  int idx = get_global_id(0);
  unsigned long n = memsize/sizeof(TYPE);
  int total_num_threads = get_global_size(0);  
  
  TYPE localp;
  for(i=idx;i < n; i+= total_num_threads){
    localp = buf[i];
    if ( (i+MOD_SZ-offset)%MOD_SZ == 0){
      if(localp != p1){
	RECORD_ERR(err_count, &buf[i], p1, localp);     
      }
    }else{
      if (localp != p2){
	RECORD_ERR(err_count, &buf[i], p2, localp);     	
      }
    }
  }

  return;

  
}

__kernel void
kernel_write(__global char* ptr, unsigned long memsize, TYPE p1)
{
  int i;  
  __global TYPE* buf = (__global TYPE*)ptr;
  int idx = get_global_id(0);
  unsigned long n = memsize/sizeof(TYPE);
  int total_num_threads = get_global_size(0);  
  
  for(i=idx;i < n; i+= total_num_threads){
    buf[i] = p1;
  }

  return;
  
}


__kernel void
kernel_readwrite(__global char* ptr, unsigned long memsize, TYPE p1, TYPE p2, 
		 volatile __global unsigned int* err_count,
		 __global unsigned long* err_addr, 
		 __global unsigned long* err_expect,
		 __global unsigned long* err_current, 
		 __global unsigned long* err_second_read)
{
  
  int i;
  __global TYPE* buf = (__global TYPE*) ptr;
  int idx = get_global_id(0);
  unsigned long n =  memsize/sizeof(TYPE);
  int total_num_threads = get_global_size(0);  
  TYPE localp;
  
  for(i=idx;i < n;i += total_num_threads){
    
    localp = buf[i];
    
    if (localp != p1){
      RECORD_ERR(err_count, &buf[i], p1, localp);      
    }
    
    buf[i] = p2;
  }  
  

}  


__kernel void
kernel_read(__global char* ptr, unsigned long memsize, TYPE p1,
	    volatile __global unsigned int* err_count,
	    __global unsigned long* err_addr, 
	    __global unsigned long* err_expect,
	    __global unsigned long* err_current, 
	    __global unsigned long* err_second_read)
{
  
  int i;
  __global TYPE* buf = (__global TYPE*) ptr;
  int idx = get_global_id(0);
  unsigned long n =  memsize/sizeof(TYPE);
  int total_num_threads = get_global_size(0);  
  TYPE localp;
  
  for(i=idx;i < n;i += total_num_threads){    
    localp = buf[i];    
    if (localp != p1){
      RECORD_ERR(err_count, &buf[i], p1, localp);      
    }    
  }  
  

}  

__kernel void
kernel7_write(__global char* ptr, unsigned long memsize)
{
  int i;  
  __global TYPE* buf = (__global TYPE*)ptr;
  int idx = get_global_id(0);
  unsigned long n = memsize/sizeof(TYPE);
  int total_num_threads = get_global_size(0);  
  int rand_data_num =BLOCKSIZE/sizeof(TYPE);
  
  for(i=idx;i < n; i+= total_num_threads){
    if (i < rand_data_num){
      continue;
    }
    buf[i] = buf[i%rand_data_num];
  }
  
  return;
  
}

__kernel void
kernel7_readwrite(__global char* ptr, unsigned long memsize, 
		 volatile __global unsigned int* err_count,
		 __global unsigned long* err_addr, 
		 __global unsigned long* err_expect,
		 __global unsigned long* err_current, 
		 __global unsigned long* err_second_read)
{
  
  int i;
  __global TYPE* buf = (__global TYPE*) ptr;
  int idx = get_global_id(0);
  unsigned long n =  memsize/sizeof(TYPE);
  int total_num_threads = get_global_size(0);  
  TYPE localp, expected;
  int rand_data_num =BLOCKSIZE/sizeof(TYPE);
  
  for(i=idx;i < n;i += total_num_threads){
    if (i < rand_data_num){
      continue;
    }
    localp = buf[i];
    expected = buf[i%rand_data_num];
    if (localp != expected){
      RECORD_ERR(err_count, &buf[i], expected, localp);      
    }
    
    buf[i] = ~expected;
  }  
  
}



__kernel void
kernel7_read(__global char* ptr, unsigned long memsize, 
	     volatile __global unsigned int* err_count,
	     __global unsigned long* err_addr, 
	     __global unsigned long* err_expect,
	     __global unsigned long* err_current, 
	     __global unsigned long* err_second_read)
{
  
  int i;
  __global TYPE* buf = (__global TYPE*) ptr;
  int idx = get_global_id(0);
  unsigned long n =  memsize/sizeof(TYPE);
  int total_num_threads = get_global_size(0);  
  TYPE localp, expected;
  int rand_data_num =BLOCKSIZE/sizeof(TYPE);
  
  for(i=idx;i < n;i += total_num_threads){
    if (i < rand_data_num){
      continue;
    }
    localp = buf[i];
    expected = ~(buf[i%rand_data_num]);
    if (localp != expected){
      RECORD_ERR(err_count, &buf[i], expected, localp);      
    }
    
  }  
  
}


//here we use 32 bit pattern
__kernel void
kernel_movinv32_write(__global char* ptr, unsigned long memsize, unsigned int pattern,
		      unsigned int lb, unsigned int sval, unsigned int offset)
{
  int i;  
  __global unsigned int* buf = (__global unsigned int*)ptr;
  int idx = get_global_id(0);
  unsigned long n = memsize/sizeof(unsigned int);
  int total_num_threads = get_global_size(0);  
  //assume total_num_threads can be devided by 32, which is true for our purpose 
  //then all memories written by this thread will have the same data
  unsigned int pat = pattern;
  unsigned int k=offset;
  for(i=0;i < idx % 32; i++){
    if (k >= 32){
      k=0;
      pat = lb;
    }else{
      pat = pat << 1;
      pat |= sval;
    }
  }
  
  for(i=idx;i < n; i+= total_num_threads){
    buf[i] = pat;
  }
  
  return;
  
}



//here we use 32 bit pattern
__kernel void
kernel_movinv32_readwrite(__global char* ptr, unsigned long memsize, unsigned int pattern,
			  unsigned int lb, unsigned int sval, unsigned int offset,
			  volatile __global unsigned int* err_count,
			  __global unsigned long* err_addr, 
			  __global unsigned long* err_expect,
			  __global unsigned long* err_current, 
			  __global unsigned long* err_second_read)
{
  int i;  
  __global unsigned int* buf = (__global unsigned int*)ptr;
  int idx = get_global_id(0);
  unsigned long n = memsize/sizeof(unsigned int);
  int total_num_threads = get_global_size(0);  
  //assume total_num_threads can be devided by 32, which is true for our purpose 
  //then all memories written by this thread will have the same data
  unsigned int pat = pattern;
  unsigned int k=offset;
  for(i=0;i < idx % 32; i++){
    if (k >= 32){
      k=0;
      pat = lb;
    }else{
      pat = pat << 1;
      pat |= sval;
    }
  }
  
  for(i=idx;i < n; i+= total_num_threads){
    unsigned int localp = buf[i];
    if (localp != pat){
      RECORD_ERR(err_count, &buf[i], pat, localp);      
    }
    buf[i] = ~pat;
  }
  
  return;
  
}



//here we use 32 bit pattern
__kernel void
kernel_movinv32_read(__global char* ptr, unsigned long memsize, unsigned int pattern,
		     unsigned int lb, unsigned int sval, unsigned int offset,
		     volatile __global unsigned int* err_count,
		     __global unsigned long* err_addr, 
		     __global unsigned long* err_expect,
		     __global unsigned long* err_current, 
		     __global unsigned long* err_second_read)
{
  int i;  
  __global unsigned int* buf = (__global unsigned int*)ptr;
  int idx = get_global_id(0);
  unsigned long n = memsize/sizeof(unsigned int);
  int total_num_threads = get_global_size(0);  
  //assume total_num_threads can be devided by 32, which is true for our purpose 
  //then all memories written by this thread will have the same data
  unsigned int pat = pattern;
  unsigned int k=offset;
  for(i=0;i < idx % 32; i++){
    if (k >= 32){
      k=0;
      pat = lb;
    }else{
      pat = pat << 1;
      pat |= sval;
    }
  }
  
  for(i=idx;i < n; i+= total_num_threads){
    unsigned int localp = buf[i];
    if (localp != ~pat){
      RECORD_ERR(err_count, &buf[i], ~pat, localp);      
    }
  }
  
  return;
  
}




__kernel void
kernel5_init(__global char* ptr, unsigned long memsize)
{
  int i;  
  __global unsigned int * buf = (__global unsigned int*)ptr;
  int idx = get_global_id(0);
  unsigned long n = memsize/64;
  int total_num_threads = get_global_size(0);  

  unsigned int p1 =1;
  unsigned int p2;
  p1 = p1 << (idx %32);
  p2 = ~p1;
  for(i=idx;i < n; i+= total_num_threads){
    buf[i*16] = p1;
    buf[i*16 + 1] = p1;
    buf[i*16 + 2] = p2;
    buf[i*16 + 3] = p2;
    buf[i*16 + 4] = p1;
    buf[i*16 + 5] = p1;
    buf[i*16 + 6] = p2;
    buf[i*16 + 7] = p2;
    buf[i*16 + 8] = p1;
    buf[i*16 + 9] = p1;
    buf[i*16 + 10] = p2;
    buf[i*16 + 11] = p2;
    buf[i*16 + 12] = p1;
    buf[i*16 + 13] = p1;
    buf[i*16 + 14] = p2;
    buf[i*16 + 15] = p2;
  }
  
  return;
  
}


__kernel void
kernel5_move(__global char* ptr, unsigned long memsize)
{
  int i, j;  
  int idx = get_global_id(0);
  unsigned long n = memsize/BLOCKSIZE;
  int total_num_threads = get_global_size(0);  
  //each thread is responsible for moving within 1 BLOCKSIZE
  unsigned int half_count = BLOCKSIZE/sizeof(unsigned int)/2;
  for(i=idx;i < n; i+= total_num_threads){
    __global unsigned int* mybuf = (__global unsigned int*)(ptr + i*BLOCKSIZE);
    __global unsigned int* mybuf_mid = (__global unsigned int*)(ptr + i*BLOCKSIZE + BLOCKSIZE/2);
    for(j=0;j < half_count;j++){
      mybuf_mid[j]=mybuf[j];
    }
    
    for(j=0;j < half_count -8; j++){
      mybuf[j+8] = mybuf_mid[j];
    }
    
    for(j=0;j < 8;j++){
      mybuf[j] = mybuf_mid[half_count - 8+j];
    }
    
  }

  return;
  
}


__kernel void
kernel5_check(__global char* ptr, unsigned long memsize,
	      volatile __global unsigned int* err_count,
	      __global unsigned long* err_addr, 
	      __global unsigned long* err_expect,
	      __global unsigned long* err_current, 
	      __global unsigned long* err_second_read)
{
  int i;  
  __global unsigned int * buf = (__global unsigned int*)ptr;
  int idx = get_global_id(0);
  unsigned long n = memsize/(2*sizeof(unsigned int));
  int total_num_threads = get_global_size(0);  
  
  for(i=idx;i < n; i+= total_num_threads){
    if (buf[2*i] != buf[2*i+1]){
      RECORD_ERR(err_count, &buf[2*i], buf[2*i+1], buf[2*i]);
    }
  }
  
  return;
  
}


__kernel void
kernel1_write(__global char* ptr, unsigned long memsize) 
{
  int i;  
  __global unsigned long* buf = (__global unsigned long*)ptr;
  int idx = get_global_id(0);
  unsigned long n = memsize/sizeof(unsigned long);
  int total_num_threads = get_global_size(0);  
  
  for(i=idx;i < n; i+= total_num_threads){
    buf[i] = (unsigned long)(buf+i);
  }
  
  return;
    
}

__kernel void
kernel1_read(__global char* ptr, unsigned long memsize,
	     volatile __global unsigned int* err_count,
	     __global unsigned long* err_addr, 
	     __global unsigned long* err_expect,
	     __global unsigned long* err_current, 
	     __global unsigned long* err_second_read)	     
{
  int i;  
  __global unsigned long* buf = (__global unsigned long*)ptr;
  int idx = get_global_id(0);
  unsigned long n = memsize/sizeof(unsigned long);
  int total_num_threads = get_global_size(0);  
  
  for(i=idx;i < n; i+= total_num_threads){
    if( buf[i] != (unsigned long)(buf+i)){
      RECORD_ERR(err_count, &buf[i], (buf+i), buf[i]);
    }
  }
  
  return;
    
}

/*FIXME: 
  when the @myp is replace by p, the write does not go through
  need more investigation on this when you have time
*/


__kernel void
kernel0_global_write(__global char* ptr, unsigned long memsize) 
{
  __global unsigned int* p = (__global unsigned int*)ptr;
  __global unsigned int* end_p = (__global unsigned int*)(ptr + memsize);
  
  unsigned int pattern = 1;
  unsigned int mask = 4;
  
  *p = pattern;
  pattern = (pattern << 1);

  while(p< end_p){
    __global unsigned int* myp = (__global unsigned int*)( ((unsigned int)ptr)|mask);

    if(myp == ptr){
      mask = (mask << 1);
      if (mask == 0){
	break;
      }
      continue;
    }
    
    if (myp >= end_p){
      break;
    }
    
    *myp = pattern;
    pattern = pattern <<1;    
    mask = (mask << 1);
    if (mask == 0){
      break;
    }
  }

  return;
}



__kernel void
kernel0_global_read(__global char* ptr, unsigned long memsize,
		    volatile __global unsigned int* err_count,
		    __global unsigned long* err_addr, 
		    __global unsigned long* err_expect,
		    __global unsigned long* err_current, 
		    __global unsigned long* err_second_read)			    
{

   __global unsigned int* p = (__global unsigned int*)ptr;
  __global unsigned int* end_p = (__global unsigned int*)(ptr + memsize);
  
  unsigned int pattern = 1;
  unsigned int mask = 4;
  
  if ( *p != ((unsigned int)pattern)){
    RECORD_ERR(err_count, p, pattern, *p);
  }
  pattern = (pattern << 1);
  
  while(p< end_p){
    p = (__global unsigned int*)( ((unsigned int)ptr)|mask);

    if(p == ptr){
      mask = (mask << 1);
      if (mask == 0){
	break;
      }
      continue;
    }
    
    if (p >= end_p){
      break;
    }
    if (*p != ((unsigned int)pattern)){
      RECORD_ERR(err_count, p, pattern, *p);
    }

    pattern = pattern <<1;    
    mask = (mask << 1);
    if (mask == 0){
      break;
    }
  }

  return;
}



//each thread is responsible for 1 BLOCKSIZE each time
__kernel void
kernel0_local_write(__global char* ptr, unsigned long memsize) 
{
  int i;  
  __global unsigned long* buf = (__global unsigned long*)ptr;
  int idx = get_global_id(0);
  unsigned long n = memsize/BLOCKSIZE;
  int total_num_threads = get_global_size(0);
  

  for(i=idx; i < n; i+= total_num_threads){
    __global unsigned long * start_p= (__global unsigned long)(ptr + i*BLOCKSIZE);
    __global unsigned long* end_p = (__global unsigned long*)(ptr + (i+1)*BLOCKSIZE);
    __global unsigned long * p =start_p;
    unsigned int pattern = 1;
    unsigned int mask = 8;
      
    *p = pattern;
    pattern = (pattern << 1);
    while(p< end_p){
      p = (__global unsigned long*)( ((unsigned long)start_p)|mask);
      
      if(p == start_p){
	mask = (mask << 1);
	if (mask == 0){
	  break;
	}
	continue;
      }
      
      if (p >= end_p){
	break;
      }
      
      *p = pattern;
      pattern = pattern <<1;    
      mask = (mask << 1);
      if (mask == 0){
	break;
      }
    }
  }
  return;
}

__kernel void
kernel0_local_read(__global char* ptr, unsigned long memsize,
		   volatile __global unsigned int* err_count,
		   __global unsigned long* err_addr, 
		   __global unsigned long* err_expect,
		   __global unsigned long* err_current, 
		   __global unsigned long* err_second_read)		   
{
  int i;  
  __global unsigned long* buf = (__global unsigned long*)ptr;
  int idx = get_global_id(0);
  unsigned long n = memsize/BLOCKSIZE;
  int total_num_threads = get_global_size(0);
  

  for(i=idx; i < n; i+= total_num_threads){
    __global unsigned long * start_p= (__global unsigned long)(ptr + i*BLOCKSIZE);
    __global unsigned long* end_p = (__global unsigned long*)(ptr + (i+1)*BLOCKSIZE);
    __global unsigned long * p =start_p;
    unsigned int pattern = 1;
    unsigned int mask = 8;
      
    if (*p != pattern){
      RECORD_ERR(err_count, p, pattern, *p);
    }
    
    pattern = (pattern << 1);
    while(p< end_p){
      p = (__global unsigned long*)( ((unsigned long)start_p)|mask);
      
      if(p == start_p){
	mask = (mask << 1);
	if (mask == 0){
	  break;
	}
	continue;
      }
      
      if (p >= end_p){
	break;
      }
      
      if (*p != pattern){
	RECORD_ERR(err_count, p, pattern, *p);
      }

      pattern = pattern <<1;    
      mask = (mask << 1);
      if (mask == 0){
	break;
      }
    }
  }
  return;
}
