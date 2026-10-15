# OpenCL int2 MoE expert GEMV driver.
#   source /swtools/intel/2026.0/oneapi-vars.sh --force   (icpx + CL headers/loader)
#   make && ./build/int2_moe_ocl --t 1
ONEAPI ?= /swtools/intel/2026.0
CXX      ?= icpx
CXXFLAGS  = -std=c++20 -O3 -I$(ONEAPI)/include -I../common
LDFLAGS   = -L$(ONEAPI)/lib -lOpenCL

BUILD  = build
TARGET = $(BUILD)/int2_moe_ocl

all: $(TARGET) $(BUILD)/int2_moe_prefill_ocl $(BUILD)/int2_moe_prefill.cl $(BUILD)/int2_moe_prefill_int8.cl $(BUILD)/int2_int8_dpas.cl $(BUILD)/int2_moe.cl $(BUILD)/int2_fp16_upcvt.cl $(BUILD)/epilogue.clh

$(TARGET): main.cpp ../common/dt16.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD)/int2_moe_prefill_ocl: prefill.cpp ../common/dt16.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD)/int2_moe_prefill.cl: int2_moe_prefill.cl | $(BUILD)
	cp $< $@

$(BUILD)/int2_moe_prefill_int8.cl: int2_moe_prefill_int8.cl | $(BUILD)
	cp $< $@

$(BUILD)/int2_int8_dpas.cl: ../int2_via_int2_x_int8_dpas/int2_int8_dpas.cl | $(BUILD)
	cp $< $@

# the kernels include the unchanged dense int2 source and its epilogue header
$(BUILD)/int2_moe.cl: int2_moe.cl | $(BUILD)
	cp $< $@

$(BUILD)/int2_fp16_upcvt.cl: ../int2_fp16_upcvt/int2_fp16_upcvt.cl | $(BUILD)
	cp $< $@

$(BUILD)/epilogue.clh: ../common/epilogue.clh | $(BUILD)
	cp $< $@

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
