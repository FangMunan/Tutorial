APP := snn_transform_h3b_worker
OBJECTS = $(BUILD_DIR)snn_transform_h3b_worker_v2.o $(BUILD_DIR)h2_kernels.o

# H3B adds an on-chip reduction / multicast protocol to the already-frozen
# H2 arithmetic kernels.  Without an explicit optimisation level GCC emits an
# ARM image that exceeds the 32-KiB ITCM region.  -Os changes only code
# generation/storage, not the frozen fixed-point arithmetic contract, and is
# therefore the appropriate baseline build for the distributed worker.
CFLAGS += -Os -ffunction-sections -fdata-sections -I../h2_aplx -I../h2f_aplx

ifdef SPINN_DIRS
include $(SPINN_DIRS)/make/app.make
else
include $(SPINN_INSTALL_DIR)/make/app.make
endif

$(BUILD_DIR)h2_kernels.o: ../h2_aplx/h2_kernels.c
	$(MKDIR) $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $<
