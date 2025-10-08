"""
Block size for PagedAttention:
- A single position requires 2 * layers * heads * 
"""

from tllm.core.request import Request

class KVCacheManager:
    def __init__(self, max_kv_bytes: int):
        self.max_kv_bytes = max_kv_bytes
        self.used_kv_bytes = 0
        
        # request_id -> bytes total 
        self.alloc_map: dict[str, int] = {}
    
    def can_admit(self, req: Request, kv_bytes_for_req: int) -> bool:
        return self.used_kv_bytes + kv_bytes_for_req <= self.max_kv_bytes
    
    def admit(self, req: Request, kv_bytes_for_req: int) -> None:
        self.used_kv_bytes += kv_bytes_for_req
        self.alloc_map[req.request_id] = kv_bytes_for_req
    
    def free(self, req_id: str) -> None:
        b = self.alloc_map.pop(req_id, 0)
        self.used_kv_bytes -= b
        if self.used_kv_bytes < 0:
            raise ValueError("Error in KV cache manager calculation")


def estimate_kv_bytes_per_request(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    max_seq_len: int,
) -> int:
    return 2 * max_seq_len * num_layers * num_kv_heads * head_dim * dtype_bytes 