# src/tllm/core/kv_cache/kv_cache_manager.py

from abc import ABC, abstractmethod
import torch

class KVCacheManager(ABC):
    @abstractmethod
    def allocate(self, request_id: str, num_tokens: int) -> bool:
        """Allocate space for tokens, return false if OOM."""
        ...
    
    @abstractmethod
    def free(self, request_id: str) -> None:
        """Free cache from a request"""
        ...

    @abstractmethod
    def get_cache_for_request(self, request_id: str) -> torch.Tensor:
        """Get cache tensor for forward pass"""
        ...
    
    @abstractmethod
    def update_cache(self, request_id: str, new_kv: tuple, num_tokens: int) -> torch.Tensor:
        """Store new KV values after forward pass"""
        ...
    
    @abstractmethod
    def can_allocate(self, num_tokens: int) -> bool:
        """Check if we have space to allocate tokens"""
        ...
    