# SpiNNakerManchester/spinnaker_tools app.make convention.
# Build in a SpiNNaker tools environment with SPINN_DIRS or SPINN_INSTALL_DIR set.

APP := snn_transform_h2
OBJECTS := $(BUILD_DIR)snn_transform_h2.o $(BUILD_DIR)h2_kernels.o

ifdef SPINN_DIRS
include $(SPINN_DIRS)/make/app.make
else
include $(SPINN_INSTALL_DIR)/make/app.make
endif
