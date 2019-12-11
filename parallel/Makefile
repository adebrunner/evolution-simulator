# NVCC is path to nvcc. Here it is assumed /usr/local/cuda is on one's PATH.

NVCC = nvcc

CUDAPATH = /usr/local/cuda

NVCCFLAGS = -I$(CUDAPATH)/include

gpu:
	$(NVCC) $(NVCCFLAGS) *.cu

clean:
	rm *.o