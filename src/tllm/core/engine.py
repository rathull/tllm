# src/tllm/core/engine.py
import asyncio
from tllm.core.request import Request

class Engine:
    def __init__(
        self, 
        gpu_memory_utilization: float = 0.9,
        decode_size_limit_for_prefill: float = 15,
        
        # Model params for KV sizing
        num_layers: int = 32,
        num_kv_heads: int = 16,
        head_dim: int = 128,
        dtype_bytes: int = 2,  # bfloat16
        total_kv_pool_bytes: int = 16 * (1024**3),  # e.g. 16 GiB
    ) -> None:
        self.gpu_memory_utilization = gpu_memory_utilization
        # WAITING status, no KV cache allocated, no tokens computed
        self.waiting_queue = asyncio.Queue()
        # Requests have KV cache allocated, either a chunked prefill or decode
        self.running_queue = asyncio.Queue()
        # KV cache manager
        
        # If we have more than this number of decodes waiting, don't start prefills
        self.decode_size_limit_for_prefill = decode_size_limit_for_prefill
        
        """
        Question: how should we handle chunked prefills? We can include the first
        chunk of a prefill request alongside the decode request of the step (assuming
        there is space to allocate the KV cache necessary for that prefill request).
        Otherwise, prioritize decode requests. 
        """

    async def add(self, request: Request) -> None:
        await self.waiting_queue.put(request)

    async def schedule(self):
        prefill_batch = []
        decode_batch = []
        
        # Add all decode steps we can afford to, prioritizing FCFS running requests
        for req in self.running_queue:
            # TODO(kv): if we can allocate space for this request, add it to the decode batch
            decode_batch.append(req)
        
        # If we can allocate space for another prefill request too, add it to the queue
        if (
            len(decode_batch) < self.decode_size_limit_for_prefill and 
            self.waiting_queue # and
            # can allocate space for a prefill request on GPU
        ):
            self.waiting_queue.append()
        
        # Only one new prefill per step
        # Prevents prefill from monopolizing compute, keeps decode ITL predictable

    async def step(self) -> None:
        # Pull from queue and add requests to scheduler
        # 
        
        # Schedule: select which requests to run in this step (decode or prefill)
        # batch = scheduler.schedule(...)
        
        # Forward pass: run model
        # tokens = mode_runner.forward(batch)
        
        # Forward pass: sample tokens
        
        # Postprocess: add sampled token IDs to Request, detokenize, check stop conditions
        # Stop conditions are if requests exceeds max_tokens or max_model_length, if sampled
        # token is EOD IS, if sampled token matches any of stop_token_ids
        
        
        # If a requests is finished, clean up (return KV cache blockS) and return output early 
        
        # Ask scheduler what to run
        # batch = scheduler.schdule(request_queue)
        # Schedule checks running queue -> if empty nothing
        # if waiting queue