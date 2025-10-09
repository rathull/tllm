# src/tllm/core/kv_cache/kv_cache_manager.py

from abc import ABC, abstractmethod
from tllm.core.request import Request
import torch

class KVCacheManager(ABC):
    @abstractmethod
    def allocate(self, request: Request, num_tokens: int) -> bool:
        """Allocate space for tokens, return false if OOM."""
        ...
    
    @abstractmethod
    def free(self, request: Request) -> None:
        """Free cache from a request"""
        ...

    @abstractmethod
    def bytes_for_tokens(self, num_tokens: int) -> int:
        """Get number of bytes to allocate num_tokens of KV cache space"""
        ...
    
    @property
    @abstractmethod
    def bytes_for_full_seq(self) -> int:
        """Get number of bytes to allocate the max sequence length"""
        ...
    
    @abstractmethod
    def get_cache_for_request(self, request: Request) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get cache tensors for a specific requests"""
        ...
    
    @abstractmethod
    def get_layer_cache(self, request: Request, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cache tensors for a specific layer"""
        ...
    
    @abstractmethod
    def update_cache(
        self, 
        request: Request, 
        new_k: list[torch.Tensor], 
        new_v: list[torch.Tensor], 
        num_tokens: int
    ) -> None:
        """Store new KV values after forward pass"""
        ...
    
    @abstractmethod
    def can_allocate(self, request: Request, num_tokens: int) -> bool:
        """Check if we have space to allocate tokens"""
        ...
    