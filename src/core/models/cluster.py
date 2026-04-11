from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


class MetricsPayload(BaseModel):
    """Internal wire payload sent from each node to the leader every metrics cycle."""
    probe: bool
    query_view: bool = False
    hostname: str
    role: Optional[str] = None
    metrics: Optional[Dict[str, Dict[str, Dict[str, Union[float, int]]]]] = None
    loaded_models: Optional[List[Tuple[str, float]]] = None
    load_models: Optional[bool] = None
    at_capacity: Optional[bool] = None
    total_ram_gb: Optional[float] = None
    free_ram_gb: Optional[float] = None
    total_vram_gb: Optional[float] = None
    free_vram_gb: Optional[float] = None
    gpu_support: Optional[bool] = None
    gpu_available: Optional[bool] = None


class WindowMetrics(BaseModel):
    """Embedding throughput and latency stats for a single time window."""
    rps: float = Field(..., description="Requests per second over this window")
    avg_ms: float = Field(..., description="Average local inference latency in milliseconds over this window")
    n: int = Field(..., description="Number of requests observed in this window")
    e2e_avg_ms: Optional[float] = Field(default=None, description="Average end-to-end latency in milliseconds (originating node only; null on pure serving nodes)")


class VectorMetrics(BaseModel):
    """Metrics for a single vector type (or model) on a node."""
    type: str = Field(..., description="Vector type (e.g. dense_model, sparse_model, wmtr, trigrams)")
    model: Optional[str] = Field(default=None, description="Model name for transformer-based types; null for rule-based types")
    revision: Optional[str] = Field(default=None, description="Model revision/commit hash; null if not specified or not applicable")
    windows: Dict[int, WindowMetrics] = Field(
        default_factory=dict,
        description="Per-window metrics keyed by window size in seconds (5, 30, 60, 300)"
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
    metrics: List[VectorMetrics] = Field(
        default_factory=list,
        description="Per-vector-type embedding metrics for this node; empty for API nodes"
    )


class ClusterView(BaseModel):
    """Latest cluster state as persisted by the encoder leader."""
    nodes: Dict[str, NodeView] = Field(
        default_factory=dict,
        description="Map of node ID to node snapshot for all nodes that have reported within the expiry window"
    )
