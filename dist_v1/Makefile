NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

main.exe: main.cpp
	$(NVCC) $(NVCC_FLAGS) $< -o $@

main_gpu.exe: kernel.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -vf *.o *.exe

