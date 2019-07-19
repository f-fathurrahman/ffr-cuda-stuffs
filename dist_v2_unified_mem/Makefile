NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

main_gpu.exe: main.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -vf *.o *.exe

