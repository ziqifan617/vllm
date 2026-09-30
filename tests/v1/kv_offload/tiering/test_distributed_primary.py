# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Addressing, completion fencing, and fail-closed distributed primary tests."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.v1.kv_offload.test_factory import _make_offloading_config
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.tiering import distributed_primary as module
from vllm.v1.kv_offload.tiering.distributed_primary import (
    DistributedPrimaryOffloadingSpec,
    DistributedPrimaryWorker,
)
from vllm.v1.kv_offload.tiering.primary_bridge import PrimaryBridge


@pytest.fixture
def nixl_agent(monkeypatch):
    import vllm.distributed.nixl_utils as nixl

    agent = MagicMock()
    agent.get_agent_metadata.return_value = b"metadata"
    agent.add_remote_agent.return_value = "rank-zero"
    agent.check_xfer_state.return_value = "DONE"
    monkeypatch.setattr(nixl, "NixlWrapper", MagicMock(return_value=agent))
    monkeypatch.setattr(nixl, "nixl_agent_config", MagicMock())
    return agent


def bridge(rank=4, **kwargs):
    options = dict(
        address=0x100000,
        length=3 * 36864,
        row_bytes=36864,
        slot_bytes=4096,
        rank=rank,
        world_size=8,
    )
    options.update(kwargs)
    result = PrimaryBridge(**options)
    result.connect({**result.metadata(), "rank": 0, "address": 0x200000})
    return result


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("operation", ["READ", "WRITE"])
def test_transfer_uses_global_rank_stride_with_row_padding(nixl_agent, rank, operation):
    transport = bridge(rank)
    transport.transfer(operation, [2, 0, 2])
    descriptors = nixl_agent.get_xfer_descs.call_args_list
    for call, base in zip(descriptors, [0x100000, 0x200000]):
        assert call.args[0] == [
            (base + c * 36864 + rank * 4096, 4096, 0) for c in [0, 2]
        ]
    assert nixl_agent.initialize_xfer.call_args.args[0] == operation
    assert nixl_agent.initialize_xfer.call_args.kwargs["backends"] == ["UCX"]
    nixl_agent.release_xfer_handle.assert_called_once()
    transport.close()


@pytest.mark.parametrize("chunks", [[-1], [3]])
def test_out_of_range_chunks_do_not_start_dma(nixl_agent, chunks):
    transport = bridge()
    with pytest.raises(ValueError, match="outside"):
        transport.transfer("READ", chunks)
    nixl_agent.initialize_xfer.assert_not_called()
    transport.close()


def test_empty_transfer_and_repeated_close_are_noops(nixl_agent):
    transport = bridge()
    transport.transfer("READ", [])
    transport.close()
    transport.close()
    nixl_agent.initialize_xfer.assert_not_called()
    nixl_agent.deregister_memory.assert_called_once()


@pytest.mark.parametrize("field", ["length", "row_bytes", "slot_bytes", "world_size"])
def test_mismatched_peer_geometry_is_rejected(nixl_agent, field):
    transport = PrimaryBridge(0x100000, 98304, 32768, 4096, 4, 8)
    remote = {**transport.metadata(), "rank": 0}
    remote[field] += 1
    with pytest.raises(ValueError, match=field):
        transport.connect(remote)
    nixl_agent.add_remote_agent.assert_not_called()
    transport.close()


@pytest.mark.parametrize("failure", ["ERR", RuntimeError("poll failed"), "timeout"])
def test_failed_dma_retains_registration_and_fences_future_work(
    nixl_agent, monkeypatch, failure
):
    transport = bridge()
    if failure == "timeout":
        nixl_agent.check_xfer_state.return_value = "PROC"
        monkeypatch.setattr(
            "vllm.v1.kv_offload.tiering.primary_bridge.time.monotonic",
            MagicMock(side_effect=[0.0, 61.0]),
        )
    elif isinstance(failure, Exception):
        nixl_agent.check_xfer_state.side_effect = failure
    else:
        nixl_agent.check_xfer_state.return_value = failure
    with pytest.raises((RuntimeError, TimeoutError)):
        transport.transfer("READ", [0])
    assert transport.failed
    with pytest.raises(RuntimeError, match="failed"):
        transport.transfer("READ", [1])
    with pytest.raises(RuntimeError, match="Cannot unmap"):
        transport.close()
    nixl_agent.release_xfer_handle.assert_not_called()
    nixl_agent.deregister_memory.assert_not_called()


def make_worker(remote=True):
    events = []
    underlying = MagicMock()

    def gpu_store(*args):
        events.append("gpu_submit")
        return True

    def gpu_load(*args):
        events.append("gpu_load")
        return True

    underlying.submit_store.side_effect = gpu_store
    underlying.wait.side_effect = lambda *a: events.append("gpu_done")
    underlying.submit_load.side_effect = gpu_load
    transport = MagicMock(failed=False)
    transport.transfer.side_effect = lambda op, chunks: events.append(op)
    wrapped = DistributedPrimaryWorker(underlying, transport, remote)
    return wrapped, events


def cpu_spec():
    return CPULoadStoreSpec([0, 2])


def test_store_ack_waits_for_gpu_then_gather():
    wrapped, events = make_worker()
    assert wrapped.submit_store(7, MagicMock(), cpu_spec())
    assert events == ["gpu_submit", "gpu_done", "WRITE"]


def test_load_scatter_precedes_gpu_read():
    wrapped, events = make_worker()
    assert wrapped.submit_load(7, cpu_spec(), MagicMock())
    assert events == ["READ", "gpu_load"]


def test_leader_local_store_remains_asynchronous():
    wrapped, events = make_worker(remote=False)
    assert wrapped.submit_store(7, MagicMock(), cpu_spec())
    assert events == ["gpu_submit"]


def test_rejected_gpu_store_does_not_gather():
    wrapped, events = make_worker()
    wrapped.worker.submit_store.side_effect = None
    wrapped.worker.submit_store.return_value = False
    assert not wrapped.submit_store(7, MagicMock(), cpu_spec())
    assert events == []


@pytest.mark.parametrize("direction", ["store", "load"])
def test_failed_bridge_cannot_ack_or_unmap(direction):
    wrapped, events = make_worker()
    wrapped.bridge.transfer.side_effect = RuntimeError("missing follower")
    with pytest.raises(RuntimeError, match="missing follower"):
        getattr(wrapped, "submit_" + direction)(7, cpu_spec(), cpu_spec())
    with pytest.raises(RuntimeError, match="failed"):
        wrapped.get_finished()
    with pytest.raises(RuntimeError, match="failed"):
        wrapped.shutdown()
    wrapped.worker.shutdown.assert_not_called()
    wrapped.worker.get_finished.assert_not_called()
    assert "gpu_load" not in events


@pytest.mark.parametrize("direction", ["store", "load"])
def test_delayed_bridge_cannot_ack_or_promote_early(direction):
    wrapped, events = make_worker()
    entered, release = Event(), Event()

    def delayed(op, chunks):
        entered.set()
        assert release.wait(5)

    wrapped.bridge.transfer.side_effect = delayed
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            getattr(wrapped, "submit_" + direction), 7, cpu_spec(), cpu_spec()
        )
        try:
            assert entered.wait(5)
            assert not future.done()
            assert "gpu_load" not in events
        finally:
            release.set()
        assert future.result(timeout=5)


@pytest.mark.parametrize("rank", range(8))
def test_spec_maps_each_logical_rank_and_connects_only_remote_workers(
    monkeypatch, rank
):
    config = _make_offloading_config(
        rank=rank, world_size=8, dcp_size=8, worker_kv_bytes_per_block=4096
    )
    spec = DistributedPrimaryOffloadingSpec(config)
    group = SimpleNamespace(rank_in_group=rank, world_size=8, cpu_group=object())
    monkeypatch.setattr(module, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        module, "in_the_same_node_as", lambda *a, **k: [True] * 4 + [False] * 4
    )
    region = MagicMock()
    region.total_size_bytes = 65536
    region_ctor = MagicMock(return_value=region)
    monkeypatch.setattr(module, "SharedOffloadRegion", region_ctor)
    transport = MagicMock(failed=False)
    transport_ctor = MagicMock(return_value=transport)
    monkeypatch.setattr(module, "PrimaryBridge", transport_ctor)
    worker = MagicMock()
    monkeypatch.setattr(module, "CPUOffloadingWorker", MagicMock(return_value=worker))

    def gather(peers, local, group):
        for r in range(8):
            peers[r] = dict(
                rank=r, length=65536, row_bytes=32768, slot_bytes=4096, world_size=8
            )

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    wrapped = spec.get_worker(MagicMock())
    assert region_ctor.call_args.kwargs["rank"] == rank
    assert wrapped is spec.get_worker(MagicMock())
    assert wrapped.remote == (rank >= 4)
    assert transport_ctor.called == (rank == 0 or rank >= 4)
    assert transport.connect.called == (rank >= 4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pp_size": 2},
        {"pcp_size": 2},
        {"data_parallel_size": 2},
        {"tp_size": 4},
        {"replicated_layout": True},
        {"rank": 8},
    ],
)
def test_spec_rejects_unqualified_topologies(kwargs):
    config = _make_offloading_config(world_size=8, **kwargs)
    with pytest.raises(ValueError, match="requires nonreplicated"):
        DistributedPrimaryOffloadingSpec(config)


def test_spec_rejects_canonical_layout():
    config = replace(_make_offloading_config(), canonical_layout=True)
    with pytest.raises(ValueError, match="requires nonreplicated"):
        DistributedPrimaryOffloadingSpec(config)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_spec_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        DistributedPrimaryOffloadingSpec(
            _make_offloading_config(extra_config={"primary_bridge_timeout_s": timeout})
        )
