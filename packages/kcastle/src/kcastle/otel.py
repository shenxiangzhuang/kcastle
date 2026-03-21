# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false
"""OpenTelemetry setup helpers for kcastle."""

from __future__ import annotations

import os
from typing import Any

import opentelemetry.trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def _create_span_exporter() -> Any:
    """Create an OTLP span exporter based on ``OTEL_EXPORTER_OTLP_PROTOCOL``.

    Defaults to ``http/protobuf``.  Set to ``grpc`` for gRPC transport.
    """
    protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    return OTLPSpanExporter()


def _create_log_exporter() -> Any:
    """Create an OTLP log exporter based on ``OTEL_EXPORTER_OTLP_PROTOCOL``."""
    protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    else:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    return OTLPLogExporter()


def configure_otel() -> Any:
    """Create and register OTLP-backed tracer and logger providers.

    The exporters read ``OTEL_EXPORTER_OTLP_ENDPOINT`` (and related
    variables like ``OTEL_EXPORTER_OTLP_INSECURE``) from the environment
    automatically — no explicit endpoint parameter is needed.

    Set ``OTEL_EXPORTER_OTLP_PROTOCOL`` to ``grpc`` or ``http/protobuf``
    (default) to select the transport.

    Returns the ``TracerProvider``.
    """
    resource = Resource.create({"service.name": "kcastle"})

    # Traces
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(_create_span_exporter()))
    opentelemetry.trace.set_tracer_provider(tracer_provider)

    # Logs (for future OTel log-based events)
    log_provider: Any = None
    try:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

        log_provider = LoggerProvider(resource=resource)
        log_provider.add_log_record_processor(BatchLogRecordProcessor(_create_log_exporter()))
        set_logger_provider(log_provider)
    except ImportError:
        pass  # Log SDK not installed; traces-only mode

    return tracer_provider, log_provider
