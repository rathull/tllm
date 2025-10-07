# server/completion_types.py
from pydantic import BaseModel

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0

class CompletionResponse(BaseModel):
    id: str
    text: str
    num_prompt_tokens: int
    num_completion_tokens: int
    num_total_tokens: int
    latency_ms: float
    ttft_ms: float | None  # Time to first token