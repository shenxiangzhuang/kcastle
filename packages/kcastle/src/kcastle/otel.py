# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false
"""OpenTelemetry setup helpers for kcastle."""

from __future__ import annotations

from typing import Any

import os

import opentelemetry.trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def _create_exporter() -> Any:
    """Create an OTLP exporter based on ``OTEL_EXPORTER_OTLP_PROTOCOL``.

    Defaults to ``http/protobuf``.  Set to ``grpc`` for gRPC transport.
    """
    protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    return OTLPSpanExporter()


def configure_otel() -> Any:
    """Create and register an OTLP exporter-backed tracer provider.

    The exporter reads ``OTEL_EXPORTER_OTLP_ENDPOINT`` (and related
    variables like ``OTEL_EXPORTER_OTLP_INSECURE``) from the environment
    automatically — no explicit endpoint parameter is needed.

    Set ``OTEL_EXPORTER_OTLP_PROTOCOL`` to ``grpc`` or ``http/protobuf``
    (default) to select the transport.
    """
    resource = Resource.create({"service.name": "kcastle"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(_create_exporter()))
    opentelemetry.trace.set_tracer_provider(provider)
    return provider