# H2F full resident block APLX.
# Uses only the low-level SpiNNaker tools at link time. The application reads
# the standard executed data-spec metadata directly from VCPU user0, so no
# SpiNNFrontEndCommon C library is required on the target build host.

APP := snn_transform_h2f
OBJECTS = $(BUILD_DIR)snn_transform_h2f.o $(BUILD_DIR)h2_kernels.o
CFLAGS += -I../h2_aplx

ifdef SPINN_DIRS
include $(SPINN_DIRS)/make/app.make
else
include $(SPINN_INSTALL_DIR)/make/app.make
endif

$(BUILD_DIR)h2_kernels.o: ../h2_aplx/h2_kernels.c
	$(CC) $(CFLAGS) -o $@ $<
