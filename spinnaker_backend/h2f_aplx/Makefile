# H2F full resident block APLX.
# Target code depends only on the same low-level SARK/spin1 toolchain already
# verified by H2E-board. The standard executed data-spec metadata is decoded
# locally from VCPU user0.

APP := snn_transform_h2f
OBJECTS = $(BUILD_DIR)snn_transform_h2f.o $(BUILD_DIR)h2_kernels.o
CFLAGS += -I../h2_aplx

ifdef SPINN_DIRS
include $(SPINN_DIRS)/make/app.make
else
include $(SPINN_INSTALL_DIR)/make/app.make
endif

$(BUILD_DIR)h2_kernels.o: ../h2_aplx/h2_kernels.c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $<
