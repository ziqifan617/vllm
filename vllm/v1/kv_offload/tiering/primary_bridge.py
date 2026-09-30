# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synchronous NIXL bridge for the experimental distributed CPU primary."""

import math
import time
import uuid
from collections.abc import Iterable
from typing import Any, Literal


class PrimaryBridge:
    """Move one logical rank's slots between node-local primary regions.

    A failed or timed-out transfer poisons the bridge. Its registration must
    remain alive until process exit: an exception does not cancel remote DMA.
    """

    def __init__(
        self,
        address: int,
        length: int,
        row_bytes: int,
        slot_bytes: int,
        rank: int,
        world_size: int,
        timeout_s: float = 60.0,
    ) -> None:
        if (
            address <= 0
            or row_bytes <= 0
            or slot_bytes <= 0
            or length <= 0
            or length % row_bytes
            or world_size <= 0
            or world_size * slot_bytes > row_bytes
            or not 0 <= rank < world_size
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("Invalid distributed primary geometry or timeout")

        from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

        if NixlWrapper is None or nixl_agent_config is None:
            raise RuntimeError("DistributedPrimaryOffloadingSpec requires NIXL/UCX")
        self.address = address
        self.length = length
        self.row_bytes = row_bytes
        self.slot_bytes = slot_bytes
        self.rank = rank
        self.world_size = world_size
        self.timeout_s = timeout_s
        self.failed = False
        self._closed = False
        self._remote_name: str | None = None
        self._remote_address = 0
        self._active: Any = None
        self._active_descriptors: tuple[Any, Any] | None = None
        self._agent = NixlWrapper(
            "primary-bridge-" + uuid.uuid4().hex,
            nixl_agent_config(backends=[], enable_prog_thread=True),
        )
        self._agent.create_backend("UCX", {"num_threads": "2"})
        self._registration = self._agent.register_memory(
            [(address, length, 0, "")], mem_type="DRAM"
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "metadata": self._agent.get_agent_metadata(),
            "address": self.address,
            "length": self.length,
            "row_bytes": self.row_bytes,
            "slot_bytes": self.slot_bytes,
            "world_size": self.world_size,
            "rank": self.rank,
        }

    def connect(self, remote: dict[str, Any]) -> None:
        if self._closed or self.failed or self._remote_name is not None:
            raise RuntimeError("Primary bridge is closed, failed, or already connected")
        for field in ("length", "row_bytes", "slot_bytes", "world_size"):
            if remote[field] != getattr(self, field):
                raise ValueError(f"Incompatible primary bridge {field}")
        if remote["rank"] != 0 or remote["address"] <= 0:
            raise ValueError("Primary bridge peer must be the rank-zero region")
        self._remote_name = self._agent.add_remote_agent(remote["metadata"])
        self._remote_address = remote["address"]

    def transfer(
        self, operation: Literal["READ", "WRITE"], chunks: Iterable[int]
    ) -> None:
        if self.failed or self._closed:
            raise RuntimeError("Primary bridge is failed or closed")
        if operation not in ("READ", "WRITE") or self._remote_name is None:
            raise ValueError("Invalid operation or missing primary peer")
        chunk_ids = sorted({int(chunk) for chunk in chunks})
        if any(c < 0 or c >= self.length // self.row_bytes for c in chunk_ids):
            raise ValueError("Chunk outside registered primary region")
        if not chunk_ids:
            return
        offsets = [c * self.row_bytes + self.rank * self.slot_bytes for c in chunk_ids]
        local = self._agent.get_xfer_descs(
            [(self.address + offset, self.slot_bytes, 0) for offset in offsets],
            mem_type="DRAM",
        )
        remote = self._agent.get_xfer_descs(
            [(self._remote_address + offset, self.slot_bytes, 0) for offset in offsets],
            mem_type="DRAM",
        )
        try:
            self._active_descriptors = (local, remote)
            self._active = self._agent.initialize_xfer(
                operation, local, remote, self._remote_name, backends=["UCX"]
            )
            self._agent.transfer(self._active)
            deadline = time.monotonic() + self.timeout_s
            while True:
                state = self._agent.check_xfer_state(self._active)
                if state == "DONE":
                    break
                if state not in ("PROC", "PEND"):
                    raise RuntimeError(f"Primary bridge {operation} failed: {state}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Primary bridge {operation} timed out")
                time.sleep(0.0001)
            self._agent.release_xfer_handle(self._active)
            self._active = None
            self._active_descriptors = None
        except Exception:
            self.failed = True
            raise

    def close(self) -> None:
        if self._closed:
            return
        if self.failed or self._active is not None:
            raise RuntimeError("Cannot unmap a failed primary bridge; exit the worker")
        if self._remote_name is not None:
            self._agent.remove_remote_agent(self._remote_name)
            self._remote_name = None
        self._agent.deregister_memory(self._registration)
        self._closed = True
