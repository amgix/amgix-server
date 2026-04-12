from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class MetricsPayload(BaseModel):
    """Internal wire payload sent from each node to the leader every metrics cycle."""
    probe: bool
    query_view: bool = False
    hostname: str
    role: Optional[str] = None
    metrics: Optional[List["NodeMetricSeries"]] = None
    loaded_models: Optional[List[Tuple[List[str], float]]] = None
    load_models: Optional[bool] = None
    at_capacity: Optional[bool] = None
    total_ram_gb: Optional[float] = None
    free_ram_gb: Optional[float] = None
    total_vram_gb: Optional[float] = None
    free_vram_gb: Optional[float] = None
    gpu_support: Optional[bool] = None
    gpu_available: Optional[bool] = None


class WindowSample(BaseModel):
    """Aggregated value and sample count for a single rolling window."""
    value: float = Field(..., description="Aggregated value (rate/sec, average, or sum depending on the metric)")
    n: int = Field(..., description="Number of samples in this window")


class NodeMetricSeries(BaseModel):
    """
    One metric stream on a node with rolling-window snapshots.

    key[0] is the metric name (e.g. 'batches', 'inference_ms', 'inference_origin_ms', 'hops').
    key[1:] are optional dimensions (e.g. vector type, model name, revision).
    windows are keyed by window size in seconds.
    """
    key: List[str] = Field(..., description="Compound key: [metric_name, *dimensions]")
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
    load_models: bool = Field(..., description="Whether this node is configured to load embedding models")
    at_capacity: bool = Field(..., description="Whether this node is currently at memory capacity and cannot load additional models")
    last_seen: float = Field(..., description="Unix timestamp of the last heartbeat received from this node")
    gpu_support: bool = Field(..., description="Whether the node was built with GPU library support")
    gpu_available: bool = Field(..., description="Whether a GPU device was detected and is usable on this node")
    total_ram_gb: Optional[float] = Field(default=None, description="Total system RAM in GB")
    free_ram_gb: Optional[float] = Field(default=None, description="Currently free system RAM in GB")
    total_vram_gb: Optional[float] = Field(default=None, description="Total GPU VRAM in GB; null on CPU-only nodes")
    free_vram_gb: Optional[float] = Field(default=None, description="Currently free GPU VRAM in GB; null on CPU-only nodes")
    loaded_models: List[str] = Field(
        default_factory=list,
        description="Names of models currently loaded on this node (format: name or name:revision)"
    )
    metrics: List[NodeMetricSeries] = Field(
        default_factory=list,
        description="Metric series for this node; empty for API nodes",
    )


class ClusterView(BaseModel):
    """Latest cluster state as persisted by the encoder leader."""
    nodes: Dict[str, NodeView] = Field(
        default_factory=dict,
        description="Map of node ID to node snapshot for all nodes that have reported within the expiry window"
    )
