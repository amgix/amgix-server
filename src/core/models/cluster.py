from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

NodeMeta = Dict[str, Any]


class MetricsPayload(BaseModel):
    """Internal wire payload sent from each node to the leader every metrics cycle."""
    probe: bool
    query_view: bool = False
    query_window: Optional[int] = None
    hostname: str
    source: Optional[str] = None
    role: Optional[str] = None
    metrics: Optional[List["NodeMetricSeries"]] = None
    meta: Optional[NodeMeta] = None


class WindowSample(BaseModel):
    """Mergeable numerator/denominator stats for a single window."""
    value: float = Field(..., description="Mergeable numerator for the window")
    n: Optional[int] = Field(default=None, description="Mergeable denominator for average-like metrics")


class MetricsBucket(BaseModel):
    """One mergeable aligned bucket that can be used for live and historical metrics."""
    key: str = Field(..., description="Metric name for this bucket")
    dims: List[str] = Field(default_factory=list, description="Optional metric dimensions for this bucket")
    bucket_start: int = Field(..., description="Unix timestamp of the inclusive bucket start boundary")
    bucket_seconds: int = Field(..., description="Bucket duration in seconds")
    value: float = Field(..., description="Mergeable numerator for the bucket")
    n: Optional[int] = Field(default=None, description="Mergeable denominator for average-like metrics")


class NodeMetricSeries(BaseModel):
    """
    One metric stream on a node with mergeable window stats.

    key is the metric name (e.g. 'batches', 'inference_ms', 'inference_origin_ms', 'hops').
    dims are optional dimensions (e.g. vector type, model name, revision).
    windows are keyed by window size in seconds.
    """
    key: str = Field(..., description="Metric name")
    dims: List[str] = Field(default_factory=list, description="Optional metric dimensions")
    windows: Dict[int, WindowSample] = Field(
        default_factory=dict,
        description="Rolling-window snapshots keyed by window size in seconds",
    )
    last_seen: Optional[float] = Field(
        default=None,
        description="UTC timestamp (time.time()) of the most recent sample for this series",
    )


class NodeView(BaseModel):
    """Snapshot of a single cluster node as last reported to the leader."""
    role: str = Field(..., description="Node role: 'index', 'query', 'all' for encoder nodes; 'api' for API nodes")
    is_leader: bool = Field(default=False, description="Whether this node is the current encoder leader")
    last_seen: float = Field(..., description="Unix timestamp of the last heartbeat received from this node")
    meta: NodeMeta = Field(
        default_factory=dict,
        description="Forward-compatible node metadata bag (for load/capacity/gpu/memory/model details and future node status values)",
    )
    metrics: List[NodeMetricSeries] = Field(
        default_factory=list,
        description="Metric series for this node; empty for API nodes",
    )


class Metrics(BaseModel):
    """Latest metrics state as persisted by the encoder leader."""
    nodes: Dict[str, NodeView] = Field(
        default_factory=dict,
        description="Map of node ID to node snapshot for all nodes that have reported within the expiry window"
    )


class MetricTrend(BaseModel):
    """Historical buckets for a single metric key at a given resolution."""
    key: str = Field(..., description="Metric key")
    bucket_seconds: int = Field(..., description="Bucket resolution in seconds")
    buckets: List[MetricsBucket] = Field(default_factory=list, description="Buckets ordered by bucket_start ascending")
