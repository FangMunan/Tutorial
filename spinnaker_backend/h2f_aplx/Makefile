# H2F full resident block APLX.
# Requires the SpiNNaker low-level tools plus installed SpiNNFrontEndCommon
# C library/headers (data_specification.h, libspinn_frontend_common.a).

APP := snn_transform_h2f
OBJECTS = $(BUILD_DIR)snn_transform_h2f.o $(BUILD_DIR)h2_kernels.o
CFLAGS += -I../h2_aplx
LDFLAGS += -lspinn_frontend_common

ifdef SPINN_DIRS
include $(SPINN_DIRS)/make/app.make
else
include $(SPINN_INSTALL_DIR)/make/app.make
endif

$(BUILD_DIR)h2_kernels.o: ../h2_aplx/h2_kernels.c
	$(CC) $(CFLAGS) -o $@ $<
