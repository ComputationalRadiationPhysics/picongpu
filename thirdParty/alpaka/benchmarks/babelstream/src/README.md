This work was initially based on the [cupla port of BabelStream](https://github.com/jyoung3131/BabelStream) from Jeff Young. Then refactored.
The benchmark BabelStream is developed by Tom Deakin, Simon McIntosh-Smith, University of Bristol HPC; based on John D. McCalpin's original STREAM benchmark for CPUs
Some implementations and the documents are accessible through https://github.com/UoB-HPC

# Example Run
Can be run with custom arguments as well as catch2 arguments
# With Custom arguments:
./babelstream  --array-size=1280000 --number-runs=10
# With Catch2 arguments:
./babelstream --success
# With Custom and catch2 arguments together:
./babelstream  --success --array-size=1280000 --number-runs=10

# Command for a benchmarking run
# ./babelstream --array-size=33554432 --number-runs=100 
# Otuput is below:

'''Array size provided: 33554432
Number of runs provided: 100
Randomness seeded to: 2775986196


AcceleratorType:AccCpuSerial<1,unsigned int>
NumberOfRuns:100
Precision:single
DataSize(items):33554432
DeviceName:13th Gen Intel(R) Core(TM) i7-1360P
WorkDivInit :{gridBlockExtent: (33554432), blockThreadExtent: (1), threadElemExtent: (1)}
WorkDivCopy :{gridBlockExtent: (33554432), blockThreadExtent: (1), threadElemExtent: (1)}
WorkDivMult :{gridBlockExtent: (33554432), blockThreadExtent: (1), threadElemExtent: (1)}
WorkDivAdd  :{gridBlockExtent: (33554432), blockThreadExtent: (1), threadElemExtent: (1)}
WorkDivTriad:{gridBlockExtent: (33554432), blockThreadExtent: (1), threadElemExtent: (1)}
Kernels         Bandwidths(GB/s) MinTime(s) MaxTime(s) AvgTime(s) DataUsage(MB) 
 InitKernel      12.2133         0.0219789 0.0244341 0.0234795 268.435 
 CopyKernel      20.8898         0.01285  0.0141298 0.0130288 268.435 
 MultKernel      20.9943         0.0127861 0.0161767 0.0129707 268.435 
 AddKernel       24.4181         0.01649  0.0178725 0.0166714 402.653 
 TriadKernel     24.44           0.0164751 0.0182611 0.0166579 402.653 



AcceleratorType:AccGpuCudaRt<1,unsigned int>
NumberOfRuns:100
Precision:single
DataSize(items):33554432
DeviceName:NVIDIA RTX A500 Laptop GPU
WorkDivInit :{gridBlockExtent: (32768), blockThreadExtent: (1024), threadElemExtent: (1)}
WorkDivCopy :{gridBlockExtent: (32768), blockThreadExtent: (1024), threadElemExtent: (1)}
WorkDivMult :{gridBlockExtent: (32768), blockThreadExtent: (1024), threadElemExtent: (1)}
WorkDivAdd  :{gridBlockExtent: (32768), blockThreadExtent: (1024), threadElemExtent: (1)}
WorkDivTriad:{gridBlockExtent: (32768), blockThreadExtent: (1024), threadElemExtent: (1)}
WorkDivDot  :{gridBlockExtent: (256), blockThreadExtent: (1024), threadElemExtent: (1)}
Kernels         Bandwidths(GB/s) MinTime(s) MaxTime(s) AvgTime(s) DataUsage(MB) 
 InitKernel      62.3725         0.00430374 0.00434411 0.00433501 268.435 
 CopyKernel      90.2948         0.00297288 0.00302862 0.00300712 268.435 
 MultKernel      90.3858         0.00296988 0.00302989 0.00300866 268.435 
 AddKernel       90.947          0.00442734 0.00448436 0.00446751 402.653 
 TriadKernel     90.88           0.0044306 0.00447952 0.00446739 402.653 
 DotKernel       93.369          0.002875 0.00291691 0.0029106 268.435 



AcceleratorType:AccCpuSerial<1,unsigned int>
NumberOfRuns:100
Precision:double
DataSize(items):33554432
DeviceName:13th Gen Intel(R) Core(TM) i7-1360P
WorkDivInit :{gridBlockExtent: (33554432), blockThreadExtent: (1), threadElemExtent: (1)}
WorkDivCopy :{gridBlockExtent: (33554432), blockThreadExtent: (1), threadElemExtent: (1)}
WorkDivMult :{gridBlockExtent: (33554432), blockThreadExtent: (1), threadElemExtent: (1)}
WorkDivAdd  :{gridBlockExtent: (33554432), blockThreadExtent: (1), threadElemExtent: (1)}
WorkDivTriad:{gridBlockExtent: (33554432), blockThreadExtent: (1), threadElemExtent: (1)}
WorkDivDot  :{gridBlockExtent: (256), blockThreadExtent: (1024), threadElemExtent: (1)}
Kernels         Bandwidths(GB/s) MinTime(s) MaxTime(s) AvgTime(s) DataUsage(MB) 
 InitKernel      12.2326         0.0438886 0.0543366 0.0463925 536.871 
 CopyKernel      20.8888         0.0257014 0.0272265 0.0260267 536.871 
 MultKernel      21.0395         0.0255173 0.0292734 0.0262349 536.871 
 AddKernel       24.6628         0.0326527 0.0383083 0.0334047 805.306 
 TriadKernel     24.5604         0.0327888 0.0494151 0.0335766 805.306 



AcceleratorType:AccGpuCudaRt<1,unsigned int>
NumberOfRuns:100
Precision:double
DataSize(items):33554432
DeviceName:NVIDIA RTX A500 Laptop GPU
WorkDivInit :{gridBlockExtent: (32768), blockThreadExtent: (1024), threadElemExtent: (1)}
WorkDivCopy :{gridBlockExtent: (32768), blockThreadExtent: (1024), threadElemExtent: (1)}
WorkDivMult :{gridBlockExtent: (32768), blockThreadExtent: (1024), threadElemExtent: (1)}
WorkDivAdd  :{gridBlockExtent: (32768), blockThreadExtent: (1024), threadElemExtent: (1)}
WorkDivTriad:{gridBlockExtent: (32768), blockThreadExtent: (1024), threadElemExtent: (1)}
WorkDivDot  :{gridBlockExtent: (256), blockThreadExtent: (1024), threadElemExtent: (1)}
Kernels         Bandwidths(GB/s) MinTime(s) MaxTime(s) AvgTime(s) DataUsage(MB) 
 InitKernel      62.4307         0.00859947 0.00864104 0.00862767 536.871 
 CopyKernel      89.4157         0.00600421 0.00607738 0.00604754 536.871 
 MultKernel      89.2831         0.00601313 0.00606791 0.0060488 536.871 
 AddKernel       90.5499         0.00889351 0.00895834 0.00893668 805.306 
 TriadKernel     90.5685         0.00889168 0.00897055 0.00893744 805.306 
 DotKernel       93.2451         0.00575763 0.00581312 0.00579143 536.871 
'''
