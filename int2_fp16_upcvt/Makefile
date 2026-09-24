# OpenCL int2 x fp16 upcvt GEMM driver.
#   source /swtools/intel/2026.0/oneapi-vars.sh --force   (icpx + CL headers/loader)
#   make && ./build/int2_fp16_upcvt_ocl --m 1 --n 17408 --k 5120
ONEAPI ?= /swtools/intel/2026.0
CXX      ?= icpx
CXXFLAGS  = -std=c++20 -O3 -fopenmp -I$(ONEAPI)/include -I../common
LDFLAGS   = -L$(ONEAPI)/lib -lOpenCL -fopenmp

BUILD  = build
TARGET = $(BUILD)/int2_fp16_upcvt_ocl

all: $(TARGET) $(BUILD)/int2_fp16_upcvt.cl $(BUILD)/epilogue.clh

$(TARGET): main.cpp ../common/dt16.hpp ../common/epilogue.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# the driver loads the kernel source (and its include) from next to the binary
$(BUILD)/int2_fp16_upcvt.cl: int2_fp16_upcvt.cl | $(BUILD)
	cp $< $@

$(BUILD)/epilogue.clh: ../common/epilogue.clh | $(BUILD)
	cp $< $@

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
