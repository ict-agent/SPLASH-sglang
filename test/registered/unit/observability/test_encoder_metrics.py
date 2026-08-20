from types import SimpleNamespace

from prometheus_client import CollectorRegistry

from sglang.srt.observability.metrics_collector import (
    EncoderMetricsCollector,
    create_encoder_metrics_collector,
)


def _server_args(enable_metrics=True, extra_metric_labels=None):
    return SimpleNamespace(
        enable_metrics=enable_metrics,
        served_model_name="glm5-next-test",
        model_path="unused",
        encoder_transfer_backend="mooncake",
        extra_metric_labels=extra_metric_labels,
    )


def _value(registry, name, labels=None):
    return registry.get_sample_value(name, labels or {})


def _base_labels(**extra):
    return {
        "model_name": "glm5-next-test",
        "dp_rank": "0",
        **extra,
    }


def _metrics(registry, extra_metric_labels=None):
    metrics = create_encoder_metrics_collector(
        _server_args(extra_metric_labels=extra_metric_labels), 0, registry
    )
    assert metrics is not None
    return metrics


def test_encoder_metric_schema_matches_upstream_and_keeps_extensions():
    registry = CollectorRegistry()
    metrics = _metrics(registry)

    assert type(metrics) is EncoderMetricsCollector
    upstream_request_labels = ("model_name", "dp_rank", "modality")
    assert metrics.requests_total._type == "counter"
    assert metrics.requests_total._labelnames == upstream_request_labels + ("status",)
    assert metrics.requests_received_total._labelnames == upstream_request_labels
    assert metrics.cache_hit_tokens_total._labelnames == upstream_request_labels
    assert metrics.cache_total_tokens_total._labelnames == upstream_request_labels
    assert metrics.cache_hit_files_total._labelnames == upstream_request_labels
    assert metrics.cache_total_files_total._labelnames == upstream_request_labels
    assert metrics.cache_evictions_total._labelnames == upstream_request_labels
    assert metrics.cache_size_mb._labelnames == ("model_name", "dp_rank")
    assert metrics.cache_entries._labelnames == ("model_name", "dp_rank")
    assert metrics.mm_items_per_batch._labelnames == upstream_request_labels
    assert metrics.mm_items_per_request._labelnames == upstream_request_labels
    assert (
        metrics.encoder_request_e2e_latency_seconds._labelnames
        == upstream_request_labels
    )
    assert metrics.queue_wait_seconds._labelnames == upstream_request_labels
    assert metrics.preprocess_seconds._labelnames == upstream_request_labels
    assert metrics.model_forward_seconds._labelnames == upstream_request_labels
    assert metrics.transfer_seconds._labelnames == (
        "model_name",
        "dp_rank",
        "backend",
    )

    assert metrics.mm_items_per_batch._upper_bounds[:-1] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        32.0,
        64.0,
        128.0,
    ]
    assert metrics.mm_items_per_request._upper_bounds[:-1] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        10.0,
        12.0,
        16.0,
        24.0,
        32.0,
        64.0,
    ]
    assert metrics.encoder_request_e2e_latency_seconds._upper_bounds[:-1] == [
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        30.0,
        60.0,
    ]

    for redundant_metric in (
        "request_latency",
        "vit_embedding_time",
        "send_embedding_latency",
        "mm_cache_total",
        "embedding_num_tokens",
        "cache_requests",
        "embedding_size",
        "transfer_requests",
        "batch_requests",
    ):
        assert not hasattr(metrics, redundant_metric)

    extension_request_labels = (
        "model_name",
        "dp_rank",
        "modality",
        "execution_path",
    )
    assert metrics.requests_in_flight._labelnames == extension_request_labels
    assert metrics.queue_depth._labelnames == ("model_name", "dp_rank")
    assert metrics.requests_processed._labelnames == extension_request_labels + (
        "outcome",
        "error_stage",
    )
    assert metrics.stage_duration._labelnames == extension_request_labels + ("stage",)
    assert metrics.embedding_tokens._labelnames == extension_request_labels
    assert metrics.transfer_bytes._labelnames == (
        "model_name",
        "dp_rank",
        "modality",
    )


def test_embedding_metric_buckets():
    registry = CollectorRegistry()
    metrics = _metrics(registry)

    metrics.observe_embedding("image", "single", 163840)
    assert metrics.embedding_tokens._upper_bounds[-4:-1] == [
        163840.0,
        196608.0,
        229376.0,
    ]


def test_request_metrics_record_canonical_and_extension_series():
    registry = CollectorRegistry()
    metrics = _metrics(registry)

    metrics.request_started("video", "single")
    assert (
        _value(
            registry,
            "sglang:encoder_requests_received_total",
            _base_labels(modality="video"),
        )
        == 1
    )
    assert (
        _value(
            registry,
            "sglang:encoder_requests_in_flight",
            _base_labels(modality="video", execution_path="single"),
        )
        == 1
    )

    metrics.observe_request_e2e_latency(0.25, "video")
    metrics.request_finished("video", "single", "failed", "transfer")
    assert (
        _value(
            registry,
            "sglang:encoder_requests_total",
            _base_labels(modality="video", status="error"),
        )
        == 1
    )
    assert (
        _value(
            registry,
            "sglang:encoder_request_e2e_latency_seconds_count",
            _base_labels(modality="video"),
        )
        == 1
    )
    assert (
        _value(
            registry,
            "sglang:encoder_requests_in_flight",
            _base_labels(modality="video", execution_path="single"),
        )
        == 0
    )
    assert (
        _value(
            registry,
            "sglang:encoder_requests_processed_total",
            _base_labels(
                modality="video",
                execution_path="single",
                outcome="failed",
                error_stage="transfer",
            ),
        )
        == 1
    )


def test_meta_only_is_excluded_from_canonical_request_metrics():
    registry = CollectorRegistry()
    metrics = _metrics(registry)

    metrics.request_started("image", "meta_only", include_canonical=False)
    metrics.request_finished("image", "meta_only", "success", include_canonical=False)

    assert (
        _value(
            registry,
            "sglang:encoder_requests_received_total",
            _base_labels(modality="image"),
        )
        is None
    )
    assert (
        _value(
            registry,
            "sglang:encoder_requests_total",
            _base_labels(modality="image", status="success"),
        )
        is None
    )
    assert (
        _value(
            registry,
            "sglang:encoder_request_e2e_latency_seconds_count",
            _base_labels(modality="image"),
        )
        is None
    )
    assert (
        _value(
            registry,
            "sglang:encoder_requests_processed_total",
            _base_labels(
                modality="image",
                execution_path="meta_only",
                outcome="success",
                error_stage="none",
            ),
        )
        == 1
    )


def test_cache_batch_and_transfer_metrics_share_upstream_samples():
    registry = CollectorRegistry()
    metrics = _metrics(registry)

    metrics.record_cache_tokens(64, 96, modality="image")
    metrics.record_cache_files(1, 3, modality="image")
    metrics.inc_cache_evictions("image", 2)
    metrics.set_cache_state(8 * 1024 * 1024, 4)
    metrics.observe_mm_items_per_batch(3, "image")
    metrics.observe_batch_duration("image", 0.5)
    metrics.observe_mm_items_per_request(2, "image")
    metrics.observe_transfer_attempt("image", 0.25, "failed", 4096)

    assert (
        _value(
            registry,
            "sglang:encoder_cache_hit_tokens_total",
            _base_labels(modality="image"),
        )
        == 64
    )
    assert (
        _value(
            registry,
            "sglang:encoder_cache_total_files_total",
            _base_labels(modality="image"),
        )
        == 3
    )
    assert (
        _value(
            registry,
            "sglang:encoder_cache_evictions_total",
            _base_labels(modality="image"),
        )
        == 2
    )
    assert _value(registry, "sglang:encoder_cache_size_mb", _base_labels()) == 8
    assert (
        _value(
            registry,
            "sglang:encoder_mm_items_per_batch_count",
            _base_labels(modality="image"),
        )
        == 1
    )
    assert (
        _value(
            registry,
            "sglang:encoder_mm_items_per_request_count",
            _base_labels(modality="image"),
        )
        == 1
    )
    assert (
        _value(
            registry,
            "sglang:encoder_stage_duration_seconds_sum",
            _base_labels(modality="image", execution_path="batch", stage="batch_total"),
        )
        == 0.5
    )
    assert (
        _value(
            registry,
            "sglang:encoder_stage_duration_seconds_count",
            _base_labels(modality="image", execution_path="batch", stage="batch_total"),
        )
        == 1
    )
    assert (
        _value(
            registry,
            "sglang:encoder_transfer_seconds_count",
            _base_labels(backend="mooncake"),
        )
        == 1
    )


def test_extra_metric_labels_and_factory_gating():
    registry = CollectorRegistry()
    metrics = _metrics(registry, extra_metric_labels={"service": "encoder-a"})
    assert metrics.requests_total._labelnames == (
        "model_name",
        "dp_rank",
        "service",
        "modality",
        "status",
    )

    assert (
        create_encoder_metrics_collector(
            _server_args(enable_metrics=False), 0, CollectorRegistry()
        )
        is None
    )
    assert (
        create_encoder_metrics_collector(_server_args(), 1, CollectorRegistry()) is None
    )
    assert isinstance(
        create_encoder_metrics_collector(_server_args(), 0, CollectorRegistry()),
        EncoderMetricsCollector,
    )
