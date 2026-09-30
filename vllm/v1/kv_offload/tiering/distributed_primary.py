# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, synchronous distributed-primary prototype for TP-only engines."""

import math
from typing import Any

import torch.distributed as dist
from typing_extensions import override

from vllm.distributed.parallel_state import get_tp_group, in_the_same_node_as
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.tiering.primary_bridge import PrimaryBridge
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

logger = init_logger(__name__)


class DistributedPrimaryWorker(OffloadingWorker):
    def __init__(
        self, worker: OffloadingWorker, bridge: PrimaryBridge | None, remote: bool
    ) -> None:
        if remote and bridge is None:
            raise ValueError("A remote primary worker requires a bridge")
        self.worker = worker
        self.bridge = bridge
        self.remote = remote
        self._failed = False

    def _check_healthy(self) -> None:
        if self._failed or (self.bridge is not None and self.bridge.failed):
            raise RuntimeError("Distributed primary transfer failed; exit the worker")

    @override
    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        self._check_healthy()
        assert isinstance(dst_spec, CPULoadStoreSpec)
        try:
            if not self.worker.submit_store(job_id, src_spec, dst_spec):
                return False
            if self.remote:
                assert self.bridge is not None
                self.worker.wait({job_id})
                self.bridge.transfer("WRITE", dst_spec.block_ids)
            return True
        except Exception:
            self._failed = True
            raise

    @override
    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        self._check_healthy()
        assert isinstance(src_spec, CPULoadStoreSpec)
        try:
            if self.remote:
                assert self.bridge is not None
                self.bridge.transfer("READ", src_spec.block_ids)
            return self.worker.submit_load(job_id, src_spec, dst_spec)
        except Exception:
            self._failed = True
            raise

    @override
    def get_finished(self) -> list[TransferResult]:
        self._check_healthy()
        return self.worker.get_finished()

    @override
    def wait(self, job_ids: set[int]) -> None:
        self._check_healthy()
        self.worker.wait(job_ids)

    @override
    def shutdown(self) -> None:
        self._check_healthy()
        if self.bridge is not None:
            # A failed bridge must retain its registration and mapping until
            # process exit; closing the CPU worker would unmap live DMA memory.
            self.bridge.close()
        self.worker.shutdown()


class DistributedPrimaryOffloadingSpec(TieringOffloadingSpec):
    """Gather complete rows for existing scheduler-owned secondary tiers.

    Rank zero exposes the complete CPU primary. Remote-node workers gather
    stores before acknowledging completion and read their slots before HBM
    promotion. Existing all-worker acknowledgements fence primary readiness.
    """

    def __init__(self, config: OffloadingConfig) -> None:
        super().__init__(config)
        p = config.parallel
        if (
            config.canonical_layout
            or self.replicated_layout
            or p.pp_size != 1
            or p.pcp_size != 1
            or p.data_parallel_size != 1
            or p.tp_size != p.world_size
            or not 0 <= p.rank < p.world_size
        ):
            raise ValueError(
                "DistributedPrimaryOffloadingSpec requires nonreplicated direct "
                "layout, TP-only offload workers, PP1/PCP1/DP1 (DCP is allowed)"
            )
        if self.num_chunks <= 0:
            raise ValueError("Distributed primary requires space for at least one row")
        self._bridge_timeout_s = float(
            self.extra_config.get("primary_bridge_timeout_s", 60.0)
        )
        if not math.isfinite(self._bridge_timeout_s) or self._bridge_timeout_s <= 0:
            raise ValueError("primary_bridge_timeout_s must be finite and positive")
        self._distributed_worker: OffloadingWorker | None = None

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._distributed_worker is None:
            self._distributed_worker = self._create_distributed_worker(kv_caches)
        return self._distributed_worker

    def _create_distributed_worker(
        self, kv_caches: CanonicalKVCaches
    ) -> OffloadingWorker:
        p = self.config.parallel
        group = get_tp_group()
        if p.rank != group.rank_in_group or p.world_size != group.world_size:
            raise ValueError("Offload ranks must match the complete TP worker group")
        same_region = in_the_same_node_as(group.cpu_group, source_rank=0)
        if len(same_region) != p.world_size or not same_region[0]:
            raise ValueError("Cannot identify the rank-zero shared-memory domain")
        region = SharedOffloadRegion(
            engine_id=self._engine_id,
            num_chunks=self.num_chunks,
            rank=p.rank,
            kv_bytes_per_chunk=self.kv_bytes_per_chunk,
            cpu_page_size=self.cpu_page_size_per_worker,
        )
        bridge: PrimaryBridge | None = None
        remote = not same_region[p.rank]
        worker: CPUOffloadingWorker | None = None
        try:
            if not all(same_region):
                # One serving agent on rank zero, one initiator per remote rank.
                # Peers sharing rank zero's mmap need no additional registration.
                if p.rank == 0 or remote:
                    bridge = PrimaryBridge(
                        region._base.data_ptr(),
                        region.total_size_bytes,
                        self.kv_bytes_per_chunk,
                        self.cpu_page_size_per_worker,
                        p.rank,
                        p.world_size,
                        self._bridge_timeout_s,
                    )
                peers: list[Any] = [None] * p.world_size
                dist.all_gather_object(
                    peers, bridge.metadata() if bridge else None, group=group.cpu_group
                )
                for rank, local in enumerate(same_region):
                    if rank == 0 or not local:
                        peer = peers[rank]
                        if peer is None or peer["rank"] != rank:
                            raise ValueError("Missing or duplicate primary shard rank")
                        for field in (
                            "length",
                            "row_bytes",
                            "slot_bytes",
                            "world_size",
                        ):
                            if peer[field] != peers[0][field]:
                                raise ValueError(
                                    f"Inconsistent primary geometry: {field}"
                                )
                if remote:
                    assert bridge is not None
                    bridge.connect(peers[0])
            worker = CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_chunks=self.num_chunks,
                mmap_region=region,
            )
            logger.warning_once(
                "DistributedPrimaryOffloadingSpec is experimental: synchronous "
                "NIXL/UCX gather/scatter, fatal transfer errors, no retry or "
                "throughput qualification."
            )
            return DistributedPrimaryWorker(worker, bridge, remote)
        except Exception:
            if bridge is not None:
                bridge.close()
            if worker is not None:
                worker.shutdown()
            else:
                region.cleanup()
            raise
