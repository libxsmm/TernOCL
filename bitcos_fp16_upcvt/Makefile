# OpenCL BITCOS ternary x fp16/bf16 GEMV driver.
#   source /swtools/intel/2026.0/oneapi-vars.sh --force   (CL headers/loader)
#   make && ./build/bitcos_fp16_upcvt_ocl --m 1 --n 4096 --k 4096 --z 0.40
ONEAPI ?= /swtools/intel/2026.0
CXX      ?= icpx
CXXFLAGS  = -std=c++20 -O3 -fopenmp -I$(ONEAPI)/include -I../common
LDFLAGS   = -L$(ONEAPI)/lib -lOpenCL -fopenmp

BUILD  = build
TARGET = $(BUILD)/bitcos_fp16_upcvt_ocl

all: $(TARGET) $(BUILD)/bitcos_fp16_upcvt.cl $(BUILD)/epilogue.clh

$(TARGET): main.cpp ../common/dt16.hpp ../common/epilogue.hpp ../common/bitcos.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# the driver loads the kernel source (and its include) from next to the binary
$(BUILD)/bitcos_fp16_upcvt.cl: bitcos_fp16_upcvt.cl | $(BUILD)
	cp $< $@

$(BUILD)/epilogue.clh: ../common/epilogue.clh | $(BUILD)
	cp $< $@

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
