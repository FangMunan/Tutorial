# SpiNNakerManchester/spinnaker_tools app.make convention.
# Build in a SpiNNaker tools environment with SPINN_DIRS and
# SPINN_INSTALL_DIR pointing at the low-level tools tree.

APP := snn_transform_h2
# Recursive expansion is intentional: app.make defines BUILD_DIR after this
# Makefile is parsed.
OBJECTS = $(BUILD_DIR)snn_transform_h2.o $(BUILD_DIR)h2_kernels.o

ifdef SPINN_DIRS
include $(SPINN_DIRS)/make/app.make
else
include $(SPINN_INSTALL_DIR)/make/app.make
endif
