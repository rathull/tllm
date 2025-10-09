# src/tllm/core/kv_cache/simple_kv_cache.py

from tllm.core.kv_cache.kv_cache_manager import KVCacheManager
import torch

from tllm.core.request import Request

class SimpleKVCache(KVCacheManager):
    def __init__(
        self,
        gpu_memory_utilization: float,
        device: torch.device,
        num_hidden_layers: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16
    ):
        self.req_to_kv_cache: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self.req_to_len: dict[str, int] = {}  # how many tokens are already written
        self.device = device
        self.dtype = dtype
      
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        self.max_memory_bytes = free_bytes * gpu_memory_utilization
        self.used_bytes = 0
          
        if self.dtype in [torch.bfloat16, torch.float16]:
            self.bytes_per_element = 2
        elif self.dtype == torch.float32:
            self.bytes_per_element = 4
        else:
            raise ValueError(f"Unsuppposed data type for KV cache: {dtype}")
        
        # TODO: should maintain KV cache in (kv_head, seq, head) dimension since attention
        # reads fix a head and stream the sequence instead THis is good for TMA/coalesced
        # global -> SMEM copies and vectorized loads on head_dim
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
    
    def bytes_for_tokens(self, num_tokens: int) -> int:
        return (
            2 * self.num_hidden_layers * 
            num_tokens * self.num_kv_heads * self.head_dim *
            self.bytes_per_element
        )
    
    def reserve_write_indices(self, request_id: str, input_len: int) -> torch.Tensor:
        start = self.req_to_len.get(request_id, 0)
        end = start + input_len
        if end > self.max_seq_len:
            raise ValueError(f"Request {request_id} has too many tokens to write")
        self.req_to_len[request_id] = end
        return torch.arange(start, end, device=self.device)
    
    @property
    def bytes_for_full_seq(self) -> int:
        return self.bytes_for_tokens(self.max_seq_len)

    def allocate(self, request_id: str, num_tokens: int) -> bool:
        """Allocate space for tokens, return false if OOM."""
        if request_id in self.req_to_kv_cache:
            return True
        
        # Otherwise we are on the first prefill request, try to allocate the entire sequence
        try:
            # (batch_size, max_seq_len, num_kv_heads, head_dim)
            shape = (1, self.max_seq_len, self.num_kv_heads, self.head_dim)
            
            # TODO: note that this allocates the full sequence at once, not just num_tokens
            self.req_to_kv_cache[request_id] = [
                (
                    torch.empty(size=shape, dtype=self.dtype, device=self.device, 
                                requires_grad=False),
                    torch.empty(size=shape, dtype=self.dtype, device=self.device, 
                                requires_grad=False),
                )
                for _ in range(self.num_hidden_layers)
            ]
            
            self.used_bytes += self.bytes_for_full_seq
            return True
        
        except torch.cuda.OutOfMemoryError as e:
            print(f"[WARNING] KV cache could not be allocated: {e}")
            return False
    
    def free(self, request_id: str) -> None:
        """Free cache from a request"""
        # Garbage collector should handle dealloc
        tensors = self.req_to_kv_cache.pop(request_id)
        if tensors is not None:
            self.used_bytes -= self.bytes_for_full_seq
            # torch.cuda.empty_cache()

    def get_cache_for_request(self, request_id: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get cache tensor for forward pass"""
        return self.req_to_kv_cache[request_id]
    
    def get_layer_cache(self, request_id: str, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.req_to_kv_cache[request_id][layer_idx]
    
    def update_cache(
        self,
        request_id: str,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        num_tokens: int
    ) -> None:
        """Store new KV values after forward pass"""
        # KV cache is already updated inside attention via index_copy_
        pass
        
    def can_allocate(self, request: Request, num_tokens: int | None) -> bool:
        """Check if we have space to allocate tokens"""
        if request.request_id in self.req_to_kv_cache:
            return True
        if request.generated_tokens > 0:
            # Since we allocate the full KV cache at once, the full cache has already
            # been allocated
            return True
        # Otherwise, we need to check if we can allocate the full KV cache
        need_bytes = self.bytes_for_full_seq
        return need_bytes + self.used_bytes <= self.max_memory_bytes
