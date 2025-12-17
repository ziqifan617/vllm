# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import io
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from minio import Minio

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class MMMeta:
    mm_hash: str
    num_token: int

    @staticmethod
    def make_meta(mm_hash, num_token) -> "MMMeta":
        return MMMeta(mm_hash=mm_hash, num_token=num_token)


@dataclass
class ObjStorageConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta]

    def __init__(self):
        self.mm_datas = []

    def add_mm_data(self, mm_data: MMMeta):
        self.mm_datas.append(mm_data)


class ObjStorageConnector(ECConnectorBase):
    """
    Object Storage Connector using MinIO for encoder cache persistence.

    This connector saves and loads encoder caches to/from MinIO object storage,
    enabling distributed encoder cache sharing across vLLM instances.

    Required configuration in ec_connector_extra_config:
        - minio_endpoint: MinIO server endpoint (e.g., "localhost:9000")
        - minio_access_key: MinIO access key
        - minio_secret_key: MinIO secret key
        - minio_bucket: Bucket name for storing encoder caches
        - minio_secure: Whether to use HTTPS (default: False)
    """

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        # mm_hash -> num_encoder_tokens
        self._mm_datas_need_loads: dict[str, int] = {}

        # Track which caches exist in MinIO (for scheduler-side has_caches check)
        self.encoder_cache_minio_set: set[str] = set()

        # Initialize MinIO client
        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is None:
            raise ValueError("ec_transfer_config must be set for ObjStorageConnector")

        # Get MinIO configuration from extra_config
        minio_endpoint = transfer_config.get_from_extra_config("minio_endpoint", "localhost:9000")
        minio_access_key = transfer_config.get_from_extra_config("minio_access_key", "minioadmin")
        minio_secret_key = transfer_config.get_from_extra_config("minio_secret_key", "minioadmin")
        self._minio_bucket = transfer_config.get_from_extra_config("minio_bucket", "vllm-encoder-cache")
        minio_secure = transfer_config.get_from_extra_config("minio_secure", False)

        # Initialize MinIO client
        self._minio_client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=minio_secure,
        )

        # Ensure bucket exists
        try:
            if not self._minio_client.bucket_exists(self._minio_bucket):
                self._minio_client.make_bucket(self._minio_bucket)
                logger.info("Created MinIO bucket: %s", self._minio_bucket)
            else:
                logger.info("Using existing MinIO bucket: %s", self._minio_bucket)
        except Exception as e:
            logger.error("Failed to initialize MinIO bucket: %s", e)
            raise

        logger.info(
            "ObjStorageConnector initialized with MinIO endpoint: %s, bucket: %s",
            minio_endpoint,
            self._minio_bucket,
        )

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        """
        Start loading the cache from the connector into vLLM's encoder cache.

        This method loads the encoder cache based on metadata provided by the scheduler.
        It downloads encoder caches from MinIO and loads them into GPU memory.
        It is called before `_gather_mm_embeddings` for the EC Connector. For EC,
        the `encoder_cache` and `mm_hash` are stored in `kwargs`.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            kwargs (dict): Additional keyword arguments for the connector.
        """

        # Get the metadata
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ObjStorageConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                "In connector.start_load_caches, but the connector metadata is None"
            )
            return

        # Load the EC for each mm data from MinIO
        for mm_data in metadata.mm_datas:
            if mm_data.mm_hash in encoder_cache:
                # Already loaded in GPU cache
                continue

            object_name = f"encoder_cache/{mm_data.mm_hash}.pt"
            try:
                # Download from MinIO
                response = self._minio_client.get_object(self._minio_bucket, object_name)
                tensor_bytes = response.read()
                response.close()
                response.release_conn()

                # Deserialize tensor
                buffer = io.BytesIO(tensor_bytes)
                tensor_cpu = torch.load(buffer, map_location="cpu", weights_only=True)
                buffer.close()

                # Move to GPU
                encoder_cache[mm_data.mm_hash] = tensor_cpu.cuda()
                logger.info(
                    "Successfully loaded encoder cache from MinIO for hash %s (size: %.2f MB)",
                    mm_data.mm_hash,
                    len(tensor_bytes) / (1024 * 1024),
                )
            except Exception as e:
                logger.error(
                    "Failed to load encoder cache from MinIO for hash %s: %s",
                    mm_data.mm_hash,
                    e,
                )
                raise

    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        """
        Save the encoder cache to the connector.

        This method saves the encoder cache from the worker's local storage
        to MinIO object storage for sharing across vLLM instances.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            mm_hash (str): The hash of the multimodal data whose cache is being saved.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        # Return if not a producer (or both producer and consumer)
        if not self.is_producer and not self.is_both:
            return

        object_name = f"encoder_cache/{mm_hash}.pt"
        try:
            # Serialize tensor to bytes
            buffer = io.BytesIO()
            tensor_cpu = encoder_cache[mm_hash].detach().cpu()
            torch.save(tensor_cpu, buffer)
            buffer.seek(0)
            tensor_bytes = buffer.getvalue()
            buffer_size = len(tensor_bytes)
            buffer.seek(0)

            # Upload to MinIO
            self._minio_client.put_object(
                self._minio_bucket,
                object_name,
                buffer,
                length=buffer_size,
                content_type="application/octet-stream",
            )
            buffer.close()

            # Track that this cache exists in MinIO
            self.encoder_cache_minio_set.add(mm_hash)

            logger.info(
                "Successfully saved encoder cache to MinIO for mm_hash %s (size: %.2f MB)",
                mm_hash,
                buffer_size / (1024 * 1024),
            )
        except Exception as e:
            logger.error(
                "Failed to save encoder cache to MinIO for mm_hash %s: %s", mm_hash, e
            )
            raise

    def has_caches(
        self,
        request: "Request",
    ) -> list[bool]:
        """
        Check if cache exists in MinIO for each mm_data of request.

        This method checks both the local tracking set and queries MinIO
        to determine if encoder caches are available.

        Args:
            request (Request): the request object.

        Returns:
            List of bool indicating whether the ith mm_data exists in cache or not.
        """
        result = []
        for feature in request.mm_features:
            mm_hash = feature.identifier

            # First check local tracking set (fast path)
            if mm_hash in self.encoder_cache_minio_set:
                result.append(True)
                continue

            # Check MinIO directly (slower, but authoritative)
            object_name = f"encoder_cache/{mm_hash}.pt"
            try:
                self._minio_client.stat_object(self._minio_bucket, object_name)
                # Object exists, update local tracking
                self.encoder_cache_minio_set.add(mm_hash)
                result.append(True)
            except Exception:
                # Object does not exist
                result.append(False)

        return result

    def update_state_after_alloc(
        self,
        request: "Request",
        index: int,
    ) -> None:
        """
        Update ECConnector state after encoder cache allocation.
        """
        mm_hash = request.mm_features[index].identifier
        num_encoder_token = request.get_num_encoder_embeds(index)
        # Insert mm_hash only if this block has not been recorded yet.
        self._mm_datas_need_loads[mm_hash] = num_encoder_token

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.
        This only builds metadata for mm_data that need to be loaded from MinIO.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.

        Returns:
            ObjStorageConnectorMetadata: metadata containing mm_data to load.
        """
        meta = ObjStorageConnectorMetadata()
        for mm_hash, num_encoder_token in self._mm_datas_need_loads.items():
            meta.add_mm_data(MMMeta.make_meta(mm_hash, num_encoder_token))
        self._mm_datas_need_loads.clear()
        return meta
