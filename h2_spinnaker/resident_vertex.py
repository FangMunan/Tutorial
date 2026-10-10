#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GraphFrontEnd vertex for one resident H2 full-model kernel.

The vertex is intentionally coarse-grained: model + calibration + all trajectory inputs
are written to SDRAM once, `h2_resident.aplx` executes without host callbacks, and one
recording block is read at the end.  This is the communication pattern we want to
benchmark before introducing multi-core partitioning.

The class follows the current SpiNNakerGraphFrontEnd custom-application pattern.  Campus
software may expose a slightly older API; if so, Codex should adapt imports / method
signatures while preserving the memory-region and one-upload/one-read contract.
"""
from __future__ import annotations

from enum import IntEnum
from pathlib import Path
import json
import numpy as np

from spinn_utilities.overrides import overrides
from pacman.model.graphs.machine import MachineVertex
from pacman.model.placements import Placement
from pacman.model.resources import VariableSDRAM

from spinn_front_end_common.abstract_models import AbstractGeneratesDataSpecification
from spinn_front_end_common.data import FecDataView
from spinn_front_end_common.interface.buffer_management import recording_utilities
from spinn_front_end_common.interface.buffer_management.buffer_models import AbstractReceiveBuffersToHost
from spinn_front_end_common.interface.ds import DataSpecificationGenerator
from spinn_front_end_common.utilities.constants import BYTES_PER_WORD, SYSTEM_BYTES_REQUIREMENT
from spinn_front_end_common.utilities.data_utils import generate_steps_system_data_region
from spinn_front_end_common.utilities.helpful_functions import locate_memory_region_for_placement
from spinnaker_graph_front_end.utilities import SimulatorVertex


class DataRegions(IntEnum):
    SYSTEM = 0
    PARAMS = 1
    MODEL = 2
    CALIBRATION = 3
    LABELS = 4
    CANVASES = 5
    RECORDING = 6


class Channels(IntEnum):
    RESULT = 0


def _pad4(raw: bytes) -> bytes:
    r = len(raw) % 4
    return raw if r == 0 else raw + b"\x00" * (4 - r)


def _as_u32(raw: bytes) -> np.ndarray:
    return np.frombuffer(_pad4(raw), dtype="<u4")


class H2ResidentVertex(
        SimulatorVertex, AbstractGeneratesDataSpecification,
        AbstractReceiveBuffersToHost):
    """One-core resident correctness baseline.

    Multi-core parallelization is intentionally deferred until this serial resident
    baseline passes numerical equivalence and exposes the real compute / DMA profile.
    """

    PARAM_WORDS = 16
    PARAM_BYTES = PARAM_WORDS * BYTES_PER_WORD

    def __init__(self, bundle_dir: Path, *, output_bytes: int = 262144,
                 profile_bytes: int = 4096, label: str = "H2 resident"):
        super().__init__(label, "h2_resident.aplx")
        self.bundle_dir = Path(bundle_dir)
        self.manifest = json.loads((self.bundle_dir / "manifest.json").read_text())
        self.model = (self.bundle_dir / self.manifest["tensor_image"]["file"]).read_bytes()
        calib_name = self.manifest.get("calibration_binary", {}).get("file", "calibration.bin")
        calib_path = self.bundle_dir / calib_name
        self.calibration = calib_path.read_bytes() if calib_path.exists() else b""
        ti = self.manifest["teacher_input"]
        self.labels = (self.bundle_dir / ti["labels_file"]).read_bytes()
        self.canvases = (self.bundle_dir / ti["canvases_file"]).read_bytes()
        self.output_bytes = int(output_bytes)
        self.profile_bytes = int(profile_bytes)

    @property
    @overrides(MachineVertex.sdram_required)
    def sdram_required(self) -> VariableSDRAM:
        fixed = (
            SYSTEM_BYTES_REQUIREMENT
            + recording_utilities.get_recording_header_size(len(Channels))
            + self.PARAM_BYTES
            + len(_pad4(self.model))
            + len(_pad4(self.calibration))
            + len(_pad4(self.labels))
            + len(_pad4(self.canvases))
        )
        # Recording storage is per-run variable SDRAM.
        return VariableSDRAM(fixed, self.output_bytes + self.profile_bytes)

    @overrides(AbstractGeneratesDataSpecification.generate_data_specification)
    def generate_data_specification(self, spec: DataSpecificationGenerator,
                                    placement: Placement) -> None:
        generate_steps_system_data_region(spec, DataRegions.SYSTEM, self)

        spec.reserve_memory_region(DataRegions.PARAMS, self.PARAM_BYTES)
        spec.reserve_memory_region(DataRegions.MODEL, len(_pad4(self.model)))
        spec.reserve_memory_region(DataRegions.CALIBRATION, max(4, len(_pad4(self.calibration))))
        spec.reserve_memory_region(DataRegions.LABELS, len(_pad4(self.labels)))
        spec.reserve_memory_region(DataRegions.CANVASES, len(_pad4(self.canvases)))
        self.generate_recording_region(
            spec, DataRegions.RECORDING,
            [self.output_bytes + self.profile_bytes])

        arch = self.manifest["architecture"]
        flags = 0
        flags |= 1 if self.manifest["contracts"].get("monotonic_reveal") else 0
        flags |= 2 if self.manifest["contracts"].get("block1_projection_feature_cache") else 0
        params = np.asarray([
            0x48325231,  # 'H2R1'
            1,
            len(self.model),
            len(self.calibration),
            len(self.labels),
            len(self.canvases),
            self.output_bytes,
            self.profile_bytes,
            int(arch["teacher_batch"]),
            int(arch["teacher_rounds"]),
            int(arch["seq"]),
            int(arch["d"]),
            int(arch["layers"]),
            int(arch["heads"]),
            int(arch["features"]),
            flags,
        ], dtype="<u4")

        spec.switch_write_focus(DataRegions.PARAMS)
        spec.write_array(params)
        spec.switch_write_focus(DataRegions.MODEL)
        spec.write_array(_as_u32(self.model))
        spec.switch_write_focus(DataRegions.CALIBRATION)
        if self.calibration:
            spec.write_array(_as_u32(self.calibration))
        else:
            spec.write_value(0)
        spec.switch_write_focus(DataRegions.LABELS)
        spec.write_array(_as_u32(self.labels))
        spec.switch_write_focus(DataRegions.CANVASES)
        spec.write_array(_as_u32(self.canvases))
        spec.end_specification()

    def read_result(self) -> bytes:
        raw, missing = self.get_recording_channel_data(Channels.RESULT)
        if missing:
            raise RuntimeError("H2 resident result recording is incomplete")
        return bytes(raw)

    @overrides(AbstractReceiveBuffersToHost.get_recorded_region_ids)
    def get_recorded_region_ids(self) -> list[int]:
        return [Channels.RESULT]

    @overrides(AbstractReceiveBuffersToHost.get_recording_region_base_address)
    def get_recording_region_base_address(self, placement: Placement) -> int:
        return locate_memory_region_for_placement(placement, DataRegions.RECORDING)
