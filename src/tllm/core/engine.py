# src/tllm/core/engine.py
from tllm.core.request import Request, RequestStatus
import torch
from tllm.core.kv_cache.simple_kv_cache import SimpleKVCache
from collections import deque

class Engine:
    def __init__(
        self, 
        gpu_memory_utilization: float = 0.9,
        decode_size_limit_for_prefill: float = 32,
        device: torch.device = torch.device("cuda:0"),

        # Model params for KV sizing
        num_layers: int = 32,
        max_seq_len: int = 2048,
        num_kv_heads: int = 16,
        head_dim: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        total_kv_pool_bytes: int = 16 * (1024**3),  # e.g. 16 GiB
    ) -> None:
        self.gpu_memory_utilization = gpu_memory_utilization
        # Waiting status, no KV cache allocated, no tokens computed
        self.waiting_queue: deque[Request] = deque()
        # Requests have KV cache allocated, either a chunked prefill or decode
        self.running_requests: deque[Request] = deque()
        
        self.kv_cache_manager = SimpleKVCache(
            gpu_memory_utilization=gpu_memory_utilization,
            device=device,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype
        )
        
        # If we have more than this number of decodes waiting, don't start prefills
        self.decode_size_limit_for_prefill = decode_size_limit_for_prefill
        
        """
        Question: how should we handle chunked prefills? We can include the first
        chunk of a prefill request alongside the decode request of the step (assuming
        there is space to allocate the KV cache necessary for that prefill request).
        Otherwise, prioritize decode requests. 
        """

    async def add(self, request: Request) -> None:
        self.waiting_queue.append(request)

    def schedule(self) -> tuple[Request | None, list[Request]]:
        prefill_request = None
        decode_batch = []
        
        # Take as many decode requests as we can from front of running queue (FCFS)
        while self.running_requests and self.kv_cache_manager.can_allocate(self.running_requests[0], 1):
            next_decode_req = self.running_requests[0]
            try:
                # TODO: make sure this is only allocating what's necessary for the next block of the KV
                # cache.
                self.kv_cache_manager.allocate(next_decode_req, 1)
                decode_batch.append(self.running_requests.popleft())
            except torch.cuda.OutOfMemoryError:
                print("[WARNING] KV cache manager estimation got cooked")
        
        # Check remaining requests for capacity to see if we can run a prefill in this step
        # Take only one prefill to prevent prefill monopolizing compute, keeps decode ITL predictable
        if (
            len(decode_batch) < self.decode_size_limit_for_prefill and 
            self.waiting_queue and
            self.kv_cache_manager.can_allocate(self.waiting_queue[0], None)
        ):
            prefill_request = self.waiting_queue.popleft()
            prefill_request.request_status = RequestStatus.RUNNING
            self.running_requests.append(prefill_request)
        
        return prefill_request, decode_batch

    def remove_finished(self) -> None:
        """Remove all finished requests from the running queue"""
        # Deallocate KV cache blocks for all finished requests
        for req in self.running_requests:
            if req.status == RequestStatus.FINISHED:
                self.kv_cache_manager.free(req)
        
        # Remove from running queue
        self.running_requests = deque(
            req for req in self.running_requests 
            if req.status != RequestStatus.FINISHED
        )

    async def step(self) -> None:
        # Select which requests to run in this batch
        prefill_request, decode_batch = self.schedule()
        
        import random
        
        if prefill_request:
            # Forward pass: run model
            # tokens = mode_runner.forward(batch)
            # 20 random prefill tokens
            
            # Forward pass: sample tokens
            prefill_request.generated_tokens.append(1)
            
            # Postprocess: add sampled token IDs to Request, check stop conditions
            # Stop conditions are if requests exceeds max_tokens or max_model_length, if sampled
            # token is EOD IS, if sampled token matches any of stop_token_ids
            if (
                len(prefill_request.generated_tokens) > prefill_request.max_tokens or 
                prefill_request.reached_stop_condition()
            ):
                prefill_request.status = RequestStatus.FINISHED
        
        for decode_req in decode_batch:
            decode_req.generated_tokens.append(random.randint(40, 100))

            # Postprocess: add sampled token IDs to Request, check stop conditions
            # Stop conditions are if requests exceeds max_tokens or max_model_length, if sampled
            # token is EOD IS, if sampled token matches any of stop_token_ids
            if (
                len(decode_req.generated_tokens) > decode_req.max_tokens or 
                decode_req.reached_stop_condition()
            ):
                decode_req.status = RequestStatus.FINISHED
        
        
        # Check if any requests exceed max_tokens
    
        
        
        # If a requests is finished, clean up (return KV cache blocks) and return output early 
        self.remove_finished()
        