# src/tllm/server/app.py
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from tllm.server.completion_types import CompletionResponse, CompletionRequest
import asyncio
import time

# Global engine
# engine: Engine | None = None
engine = None
engine_task: asyncio.Task | None = None
running = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engine and scheduler on startup"""
    
    global engine, engine_task, running
    
    # Startup logic
    print(f"Initializing engine...")
    
    time.sleep(0.5)
    
    print(f"Engine initialized and running: {running}")
    
    yield
    
    # Shutdown logic
    running = False
    
    if engine_task:
        await engine_task
        
    print("Engine has been shut down")

app = FastAPI(title="tLLM Inference Server", lifespan=lifespan)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "engine_initialized": engine is not None,
        "engine_running": running,
    }

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Main completion endpoint"""
    # if engine is None:
    #     raise HTTPException(status_code=503, detail="Engine not initialized")
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Mock tokenizer (split on spaces)
    prompt_tokens = request.prompt.split()
    prompt_token_ids = list(hash(token)%1000 for token in prompt_tokens)
    
    # Create request object 
    req = None
    # Add request to engine
    # engine.add(request(req))
    
    # Wait for completion (simple polling)
    # while req.status != RequestStatus.FINISHED:
    #     await asyncio.sleep(0.1)
    
    end_time = time.time()
    
    # Mock decode (just use token IDs as text)
    output_text = f"these are the decoded tokens in the completion for {request_id}"
    
    # Calculate metrics
    latency_ms = (end_time - start_time) * 1000
    tftt_ms = None
    # if req.first_token_time:
    #     tftt_ms = (req.first_token_time - req.arrival_time) * 1000
    tftt_ms = (end_time - start_time) * 19
    
    return CompletionResponse(
        id=request_id,
        text=output_text,
        num_prompt_tokens=len(prompt_token_ids),
        num_completion_tokens=30,#len(req.generated_tokens),
        num_total_tokens=len(prompt_token_ids) + 30,#len(req.generated_tokens),
        latency_ms=latency_ms,
        ttft_ms=tftt_ms
    )

@app.get("/stats")
async def get_stats():
    """Get engine performance stats"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return engine.get_stats()

def main():
    """Entry point for the CLI command"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
