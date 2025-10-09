# core/request.py
from dataclasses import dataclass, field
from enum import Enum

class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"

@dataclass
class Request:
    """A single inference request"""
    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    
    # Sampling params
    temperature: float
    top_p: float
    
    # Stop conditions
    eos_id: int | None = None
    stop_token_ids: list[int] | None = None
    
    # Status
    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: list[int] = field(default_factory=list)
    
    # Tracking prefill progress (TODO(rathull): for chunked prefill later)
    num_computed_tokens: int = 0
    
    # Timestamps for metrics
    arrival_time: float = 0.0
    first_token_time: float | None = None
    finish_time: float | None = None
    
    def reached_stop_condition(self) -> bool:
        return (
            len(self.prompt_token_ids) + len(self.generated_tokens) >= self.max_tokens or
            self.generated_tokens[-1] == self.eos_id or 
            self.generated_tokens[-1] in self.stop_token_ids
        )

@dataclass
class SchedulerOutput:
    prefill_request: list[Request]
    decode_requests: list[Request]
    
    @property
    def is_empty(self) -> bool:
        return len(self.prefill_request) == 0 and len(self.decode_requests) == 0

    def __len__(self) -> int:
        return len(self.prefill_request) + len(self.decode_requests)