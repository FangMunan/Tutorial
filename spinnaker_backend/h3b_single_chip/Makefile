APP := snn_transform_h3b_worker
OBJECTS = $(BUILD_DIR)snn_transform_h3b_worker.o $(BUILD_DIR)h2_kernels.o
CFLAGS += -I../h2_aplx -I../h2f_aplx

ifdef SPINN_DIRS
include $(SPINN_DIRS)/make/app.make
else
include $(SPINN_INSTALL_DIR)/make/app.make
endif

$(BUILD_DIR)h2_kernels.o: ../h2_aplx/h2_kernels.c
	$(MKDIR) $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $<
