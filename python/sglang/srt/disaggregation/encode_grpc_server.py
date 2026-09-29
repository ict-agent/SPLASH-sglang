"""
gRPC Encoder Server for SGLang EPD (Encode-Prefill-Decode) mode.

This server provides gRPC-based encoding for multimodal inputs.

Usage:
    python -m sglang.launch_server --model-path <model> --encoder-only --grpc-mode
"""

import asyncio
import logging
import multiprocessing as mp
import traceback
from concurrent import futures
from typing import List

import grpc
import zmq
import zmq.asyncio
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

from sglang.srt.disaggregation.encode_server import (
    MMEncoder,
    handle_scheduler_receive_url_request,
    launch_encoder,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import random_uuid
from sglang.srt.utils.network import NetworkAddress, get_zmq_socket

logger = logging.getLogger(__name__)
SGLangEncoderServicer = sglang_encoder_pb2_grpc.SglangEncoderServicer
add_SGLangEncoderServicer_to_server = (
    sglang_encoder_pb2_grpc.add_SglangEncoderServicer_to_server
)


class EncoderHealthServicer(health_pb2_grpc.HealthServicer):
    """
    Standard gRPC health check service for encoder server.
    Implements grpc.health.v1.Health for Kubernetes probes.
    """

    OVERALL_SERVER = ""
    ENCODER_SERVICE = "sglang.grpc.encoder.SglangEncoder"

    def __init__(self):
        self._serving = False

    def set_serving(self):
        self._serving = True

    def set_not_serving(self):
        self._serving = False

    async def Check(self, request, context) -> health_pb2.HealthCheckResponse:
        if self._serving:
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.SERVING
            )
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.NOT_SERVING
        )

    async def Watch(self, request, context):
        yield await self.Check(request, context)


class SGLangEncoderServer(SGLangEncoderServicer):
    """
    gRPC service implementation for SGLang encoder.
    """

    def __init__(
        self,
        encoder: MMEncoder,
        send_sockets: List[zmq.Socket],
        server_args: ServerArgs,
    ):
        self.encoder = encoder
        self.send_sockets = send_sockets
        self.server_args = server_args

    async def Encode(
        self, request: sglang_encoder_pb2.EncodeRequest, context
    ) -> sglang_encoder_pb2.EncodeResponse:
        try:
            request_dict = {
                "mm_items": list(request.mm_items),
                "req_id": request.req_id,
                "num_parts": request.num_parts,
                "part_idx": request.part_idx,
                # The rank-side run_encoder() parses the modality from this
                # dict (Modality.from_str(request["modality"])); omitting it
                # crashes every rank process on its first request when
                # tp_size > 1.
                "modality": Modality.IMAGE.name,
            }
            for socket in self.send_sockets:
                await socket.send_pyobj(request_dict)

            # gRPC encode is image-only; encoder.encode() requires modality
            (
                nbytes,
                embedding_len,
                embedding_dim,
                error_msg,
                error_code,
            ) = await self.encoder.encode(
                mm_items=list(request.mm_items),
                modality=Modality.IMAGE,
                req_id=request.req_id,
                num_parts=request.num_parts,
                part_idx=request.part_idx,
            )
            if error_msg is not None:
                # encoder.encode() stored an error EmbeddingData for this
                # req_id; drop it or it leaks until process exit (the gRPC
                # serve path does not start the orphan sweeper otherwise).
                self.encoder.embedding_to_send.pop(request.req_id, None)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)
                return sglang_encoder_pb2.EncodeResponse()

            if self.server_args.encoder_transfer_backend == "mooncake":
                return sglang_encoder_pb2.EncodeResponse(
                    embedding_size=nbytes,
                    embedding_len=embedding_len,
                    embedding_dim=embedding_dim,
                )
            elif self.server_args.encoder_transfer_backend == "zmq_to_scheduler":
                embedding_ports = list(request.embedding_port)
                logger.info(f"embedding_port = {embedding_ports}")
                if not embedding_ports:
                    await self.encoder.send_with_url(req_id=request.req_id)
                else:
                    tasks = []
                    for embedding_port in embedding_ports:
                        tasks.append(
                            self.encoder.send(
                                req_id=request.req_id,
                                prefill_host=request.prefill_host,
                                embedding_port=embedding_port,
                            )
                        )
                    await asyncio.gather(*tasks)
                    self.encoder.embedding_to_send.pop(request.req_id, None)
                return sglang_encoder_pb2.EncodeResponse()
            elif self.server_args.encoder_transfer_backend == "zmq_to_tokenizer":
                embedding_port = (
                    request.embedding_port[0] if request.embedding_port else 0
                )
                await self.encoder.send(
                    req_id=request.req_id,
                    prefill_host=request.prefill_host,
                    embedding_port=embedding_port,
                )
                self.encoder.embedding_to_send.pop(request.req_id, None)
                return sglang_encoder_pb2.EncodeResponse()

            return sglang_encoder_pb2.EncodeResponse()

        except Exception as e:
            logger.error(f"Encode error: {e}")
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return sglang_encoder_pb2.EncodeResponse()

    async def Send(
        self, request: sglang_encoder_pb2.SendRequest, context
    ) -> sglang_encoder_pb2.SendResponse:
        try:
            buffer_address = request.buffer_address if request.buffer_address else None
            mm_data = await self.encoder.send(
                req_id=request.req_id,
                prefill_host=request.prefill_host,
                embedding_port=request.embedding_port,
                session_id=request.session_id if request.session_id else None,
                buffer_address=buffer_address,
                # A bufferless /send comes from a decode-role receiver that
                # wants only the metadata frame (no RDMA transfer target).
                # Mirrors the HTTP decode path, where the encoder server-pushes
                # metadata for role=="decode".
                meta_only=buffer_address is None,
            )
            self.encoder.embedding_to_send.pop(request.req_id, None)
            if not mm_data:
                if self.encoder.metrics is not None:
                    self.encoder.metrics.inc_send_reclaimed()
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("embedding already reclaimed (no /send within TTL)")
            elif getattr(mm_data, "error_msg", None) is not None:
                # RDMA write failed: surface it on the RPC itself, not only via
                # the ZMQ error frame (whose consumer may already be gone).
                if self.encoder.metrics is not None:
                    self.encoder.metrics.inc_rdma_write_failures("transfer")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(mm_data.error_msg)
            return sglang_encoder_pb2.SendResponse()

        except Exception as e:
            logger.error(f"Send error: {e}")
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return sglang_encoder_pb2.SendResponse()

    async def SchedulerReceiveUrl(
        self, request: sglang_encoder_pb2.SchedulerReceiveUrlRequest, context
    ) -> sglang_encoder_pb2.SchedulerReceiveUrlResponse:
        try:
            await handle_scheduler_receive_url_request(
                {
                    "req_id": request.req_id,
                    "receive_count": request.receive_count,
                    "receive_url": request.receive_url,
                }
            )
            return sglang_encoder_pb2.SchedulerReceiveUrlResponse()

        except Exception as e:
            logger.error(f"SchedulerReceiveUrl error: {e}")
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return sglang_encoder_pb2.SchedulerReceiveUrlResponse()


async def serve_grpc_encoder(server_args: ServerArgs):
    ctx = mp.get_context("spawn")
    zmq_ctx = zmq.asyncio.Context(10)
    ipc_path_prefix = random_uuid()
    port_args = PortArgs.init_new(server_args)

    if server_args.dist_init_addr:
        na = NetworkAddress.parse(server_args.dist_init_addr)
        dist_init_method = na.to_tcp()
    else:
        dist_init_method = NetworkAddress(
            server_args.host or "127.0.0.1", port_args.nccl_port
        ).to_tcp()

    send_sockets: List[zmq.Socket] = []
    for rank in range(1, server_args.tp_size):
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"
        send_sockets.append(
            get_zmq_socket(zmq_ctx, zmq.PUSH, schedule_path, bind=False)
        )
        ctx.Process(
            target=launch_encoder,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        ).start()

    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
        ],
    )

    health_servicer = EncoderHealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    encoder_servicer = SGLangEncoderServer(
        encoder=encoder,
        send_sockets=send_sockets,
        server_args=server_args,
    )
    add_SGLangEncoderServicer_to_server(encoder_servicer, server)

    SERVICE_NAMES = (
        sglang_encoder_pb2.DESCRIPTOR.services_by_name["SglangEncoder"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    listen_addr = NetworkAddress(server_args.host, server_args.port).to_host_port_str()
    server.add_insecure_port(listen_addr)

    await server.start()
    logger.info(f"gRPC encoder server listening on {listen_addr}")

    # The orphan sweeper is otherwise only started by the FastAPI lifespan;
    # without it, every entry parked in embedding_to_send by a P crash /
    # Encode error / 60s-deadline expiry leaks for the process lifetime.
    sweeper_task = asyncio.create_task(encoder._sweep_stale_embeddings_loop())
    logger.info("gRPC encoder: orphan-embedding sweeper started")

    health_servicer.set_serving()

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC encoder server...")
        health_servicer.set_not_serving()
        await server.stop(grace=5)
    finally:
        sweeper_task.cancel()
