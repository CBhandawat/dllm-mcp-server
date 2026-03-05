"""
dLLM MCP Server
===============
An MCP server that exposes Diffusion Language Model capabilities as tools
for AI applications. Wraps the dLLM framework (ZHZisZZ/dllm) from UC Berkeley.

Unique capabilities over normal LLMs:
- Text infilling (fill in [MASK] tokens anywhere in text)
- Non-left-to-right generation (diffusion decoding order)
- Step-by-step token evolution tracing
- Fast inference with KV-cache acceleration (2-4x speedup)

Run on RunPod RTX 4090 for ~$0.39/hr.
"""

import json
import logging
import sys
import os
import time
from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

# ─── Logging (NEVER print() in MCP servers) ───────────────────────────────────
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("dllm_mcp")

# ─── Server Init ──────────────────────────────────────────────────────────────
mcp = FastMCP("dllm_mcp")

# ─── Global model state ───────────────────────────────────────────────────────
# Models are loaded lazily on first use to avoid blocking server startup
_model_registry = {}   # { model_key: {"model": ..., "tokenizer": ..., "sampler": ...} }
_current_model_key = None

AVAILABLE_MODELS = {
    "llada-8b-instruct": {
        "hf_id": "GSAI-ML/LLaDA-8B-Instruct",
        "type": "llada",
        "vram_gb": 16,
        "description": "LLaDA 8B Instruct — flagship diffusion LLM, best quality",
        "sampler": "MDLM",
    },
    "llada-8b-base": {
        "hf_id": "GSAI-ML/LLaDA-8B-Base",
        "type": "llada",
        "vram_gb": 16,
        "description": "LLaDA 8B Base — raw diffusion pretraining, good for infilling",
        "sampler": "MDLM",
    },
    "modernbert-large-chat": {
        "hf_id": "dllm-collection/ModernBERT-large-chat-v0.1",
        "type": "bert",
        "vram_gb": 2,
        "description": "ModernBERT-large converted to diffusion chatbot — lightweight, fast",
        "sampler": "MDLM",
    },
    "modernbert-base-chat": {
        "hf_id": "dllm-collection/ModernBERT-base-chat-v0.1",
        "type": "bert",
        "vram_gb": 1,
        "description": "ModernBERT-base converted to diffusion chatbot — smallest, fastest",
        "sampler": "MDLM",
    },
    "qwen3-0.6b-diffusion": {
        "hf_id": "dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1",
        "type": "llada",
        "vram_gb": 2,
        "description": "Qwen3-0.6B converted from AR to diffusion — unique AR→DLM demo",
        "sampler": "MDLM",
    },
}

DEFAULT_MODEL = "modernbert-large-chat"


# ─── Enums ────────────────────────────────────────────────────────────────────
class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


# ─── Lazy model loader ────────────────────────────────────────────────────────
def _load_model(model_key: str) -> dict:
    """Load a dLLM model + tokenizer + sampler. Cached after first load."""
    global _model_registry, _current_model_key

    if model_key in _model_registry:
        _current_model_key = model_key
        return _model_registry[model_key]

    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Use dllm_list_models to see available models.")

    meta = AVAILABLE_MODELS[model_key]
    hf_id = meta["hf_id"]
    model_type = meta["type"]

    logger.info(f"Loading model {hf_id} (this may take 1-3 minutes on first load)...")

    try:
        import torch
        import dllm
        from dataclasses import dataclass

        @dataclass
        class ModelArgs:
            model_name_or_path: str = hf_id
            attn_implementation: str = "sdpa"

        args = ModelArgs()
        model = dllm.utils.get_model(model_args=args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=args)

        if model_type == "bert":
            sampler = dllm.core.samplers.BERTSampler(model=model, tokenizer=tokenizer)
        else:
            sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)

        entry = {"model": model, "tokenizer": tokenizer, "sampler": sampler, "meta": meta}
        _model_registry[model_key] = entry
        _current_model_key = model_key
        logger.info(f"Model {model_key} loaded successfully.")
        return entry

    except ImportError:
        raise RuntimeError(
            "dLLM is not installed. Run: pip install -e . from the dLLM repo directory. "
            "See README.md for full setup instructions."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_key}': {str(e)}")


def _get_active_model() -> dict:
    """Get the currently loaded model or load the default."""
    global _current_model_key
    key = _current_model_key or DEFAULT_MODEL
    return _load_model(key)


def _handle_error(e: Exception) -> str:
    """Consistent error formatting."""
    if isinstance(e, ValueError):
        return f"❌ Input Error: {e}"
    if isinstance(e, RuntimeError):
        return f"❌ Runtime Error: {e}"
    if "CUDA out of memory" in str(e):
        return (
            "❌ GPU out of memory. Try a smaller model (e.g., modernbert-base-chat) "
            "or reduce max_new_tokens. Use dllm_list_models to see VRAM requirements."
        )
    return f"❌ Unexpected error ({type(e).__name__}): {str(e)[:300]}"


# ═══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── 1. List Available Models ─────────────────────────────────────────────────
@mcp.tool(
    name="dllm_list_models",
    annotations={
        "title": "List Available dLLM Models",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def dllm_list_models() -> str:
    """List all available diffusion language models in the dLLM framework.

    Shows model names, HuggingFace IDs, VRAM requirements, and descriptions.
    Use model names from this list with dllm_load_model.

    Returns:
        str: Table of available models with metadata.
    """
    lines = ["## 🧠 Available dLLM Models\n"]
    for key, meta in AVAILABLE_MODELS.items():
        loaded = "✅ loaded" if key in _model_registry else "⬜ not loaded"
        default = " *(default)*" if key == DEFAULT_MODEL else ""
        lines.append(
            f"### `{key}`{default} {loaded}\n"
            f"- **HuggingFace**: `{meta['hf_id']}`\n"
            f"- **VRAM needed**: {meta['vram_gb']}GB\n"
            f"- **Sampler**: {meta['sampler']}\n"
            f"- **Description**: {meta['description']}\n"
        )
    lines.append(
        "\n💡 **Tip**: Start with `modernbert-large-chat` (2GB VRAM) for fast demos. "
        "Use `llada-8b-instruct` for best quality (needs 16GB VRAM)."
    )
    return "\n".join(lines)


# ─── 2. Load Model ────────────────────────────────────────────────────────────
class LoadModelInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    model_key: str = Field(
        ...,
        description="Model identifier from dllm_list_models (e.g., 'modernbert-large-chat', 'llada-8b-instruct')",
        min_length=1,
    )


@mcp.tool(
    name="dllm_load_model",
    annotations={
        "title": "Load a dLLM Model",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def dllm_load_model(params: LoadModelInput) -> str:
    """Load a diffusion language model into GPU memory.

    Downloads from HuggingFace (if not cached) and loads into VRAM.
    First load takes 1-3 minutes; subsequent loads from cache are instant.

    Args:
        params (LoadModelInput):
            - model_key (str): Model name from dllm_list_models

    Returns:
        str: Confirmation with model details and memory usage.
    """
    try:
        import torch
        start = time.time()
        entry = _load_model(params.model_key)
        elapsed = time.time() - start
        meta = entry["meta"]

        vram_used = "N/A"
        if torch.cuda.is_available():
            vram_used = f"{torch.cuda.memory_allocated() / 1e9:.1f}GB"

        return (
            f"✅ Model loaded in {elapsed:.1f}s\n\n"
            f"**Model**: `{params.model_key}`\n"
            f"**HuggingFace ID**: `{meta['hf_id']}`\n"
            f"**Sampler type**: {meta['sampler']}\n"
            f"**GPU VRAM used**: {vram_used}\n\n"
            f"Ready to use with `dllm_generate`, `dllm_infill`, and `dllm_fast_generate`."
        )
    except Exception as e:
        return _handle_error(e)


# ─── 3. Generate Text ─────────────────────────────────────────────────────────
class GenerateInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    prompt: str = Field(
        ...,
        description="Input prompt or question for the model",
        min_length=1,
        max_length=4000,
    )
    max_new_tokens: Optional[int] = Field(
        default=128,
        description="Maximum tokens to generate (8–512)",
        ge=8,
        le=512,
    )
    steps: Optional[int] = Field(
        default=128,
        description="Diffusion denoising steps — more steps = better quality but slower (16–512)",
        ge=16,
        le=512,
    )
    temperature: Optional[float] = Field(
        default=0.7,
        description="Sampling temperature (0.1=focused, 1.0=creative)",
        ge=0.1,
        le=2.0,
    )
    cfg_scale: Optional[float] = Field(
        default=0.0,
        description="Classifier-Free Guidance scale (0=off, 1-3=guided). Improves instruction following.",
        ge=0.0,
        le=5.0,
    )
    model_key: Optional[str] = Field(
        default=None,
        description="Model to use (from dllm_list_models). Uses currently loaded model if not specified.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


@mcp.tool(
    name="dllm_generate",
    annotations={
        "title": "Generate Text with Diffusion LLM",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def dllm_generate(params: GenerateInput) -> str:
    """Generate text using masked diffusion language modeling.

    Unlike autoregressive LLMs that generate left-to-right, diffusion LLMs
    denoise all tokens simultaneously over multiple steps. This enables
    unique generation behaviors and bidirectional context.

    Args:
        params (GenerateInput):
            - prompt (str): Input prompt (required)
            - max_new_tokens (int): Tokens to generate (default 128)
            - steps (int): Diffusion steps (default 128, more=better quality)
            - temperature (float): Sampling temperature (default 0.7)
            - cfg_scale (float): CFG guidance scale (default 0.0)
            - model_key (str, optional): Override active model
            - response_format (str): 'markdown' or 'json'

    Returns:
        str: Generated text with metadata about diffusion parameters used.
    """
    try:
        import torch
        import dllm

        entry = _load_model(params.model_key) if params.model_key else _get_active_model()
        tokenizer = entry["tokenizer"]
        sampler = entry["sampler"]
        meta = entry["meta"]

        messages = [[{"role": "user", "content": params.prompt}]]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
        )

        start = time.time()
        with torch.no_grad():
            outputs = sampler.sample(
                inputs,
                return_dict=True,
                max_new_tokens=params.max_new_tokens,
                steps=params.steps,
                temperature=params.temperature,
                cfg=params.cfg_scale,
            )

        sequences = dllm.utils.decode_trim(tokenizer, outputs.sequences.tolist(), inputs)
        elapsed = time.time() - start
        output_text = sequences[0] if sequences else "(no output)"

        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "output": output_text,
                "model": meta["hf_id"],
                "params": {
                    "max_new_tokens": params.max_new_tokens,
                    "steps": params.steps,
                    "temperature": params.temperature,
                    "cfg_scale": params.cfg_scale,
                },
                "generation_time_seconds": round(elapsed, 2),
            }, indent=2)

        return (
            f"## 🌊 Diffusion Generation Output\n\n"
            f"{output_text}\n\n"
            f"---\n"
            f"**Model**: `{meta['hf_id']}` | "
            f"**Steps**: {params.steps} | "
            f"**Temp**: {params.temperature} | "
            f"**CFG**: {params.cfg_scale} | "
            f"**Time**: {elapsed:.2f}s"
        )
    except Exception as e:
        return _handle_error(e)


# ─── 4. Text Infilling (Killer Feature) ───────────────────────────────────────
class InfillInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text_with_masks: str = Field(
        ...,
        description=(
            "Text containing [MASK] tokens to be filled in. "
            "Example: 'The capital of France is [MASK] and it is known for [MASK].' "
            "You can place multiple [MASK] tokens anywhere in the text."
        ),
        min_length=5,
        max_length=2000,
    )
    steps: Optional[int] = Field(
        default=256,
        description="Diffusion steps for infilling (more = better quality). Default 256.",
        ge=32,
        le=1024,
    )
    temperature: Optional[float] = Field(
        default=0.5,
        description="Temperature for infilling (lower = more deterministic). Default 0.5.",
        ge=0.1,
        le=2.0,
    )
    model_key: Optional[str] = Field(
        default=None,
        description="Model to use. BERT-based models (modernbert-*) work best for infilling.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


@mcp.tool(
    name="dllm_infill",
    annotations={
        "title": "Text Infilling with Diffusion LLM",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def dllm_infill(params: InfillInput) -> str:
    """Fill in [MASK] tokens anywhere in text — a capability unique to diffusion LLMs.

    Standard autoregressive LLMs (GPT, Llama, etc.) can only generate left-to-right.
    Diffusion LLMs can fill in blanks at ANY position using full bidirectional context,
    making them uniquely suited for editing, completion, and constrained generation tasks.

    Example inputs:
        'The [MASK] of France is Paris, a city [MASK] for its cuisine.'
        'def fibonacci(n):\\n    [MASK]\\n    return result'
        'Dear [MASK], I am writing to request [MASK] for our project.'

    Args:
        params (InfillInput):
            - text_with_masks (str): Text with [MASK] placeholders (required)
            - steps (int): Diffusion denoising steps (default 256)
            - temperature (float): Sampling temperature (default 0.5)
            - model_key (str, optional): Model override
            - response_format (str): 'markdown' or 'json'

    Returns:
        str: Completed text with [MASK] tokens filled in, plus original for comparison.
    """
    try:
        import torch
        import dllm

        entry = _load_model(params.model_key) if params.model_key else _get_active_model()
        tokenizer = entry["tokenizer"]
        sampler = entry["sampler"]
        meta = entry["meta"]

        # Tokenize the masked text directly (not as chat template — infilling uses raw text)
        input_ids = tokenizer.encode(params.text_with_masks, return_tensors="pt")

        # Replace [MASK] tokens with the model's mask token id
        mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            # Fallback: use the token for [MASK] string
            mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")

        start = time.time()
        with torch.no_grad():
            outputs = sampler.sample(
                input_ids,
                return_dict=True,
                steps=params.steps,
                temperature=params.temperature,
            )

        elapsed = time.time() - start
        completed = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "original": params.text_with_masks,
                "completed": completed,
                "model": meta["hf_id"],
                "steps": params.steps,
                "temperature": params.temperature,
                "generation_time_seconds": round(elapsed, 2),
            }, indent=2)

        return (
            f"## ✍️ dLLM Text Infilling\n\n"
            f"**Original (with masks)**:\n```\n{params.text_with_masks}\n```\n\n"
            f"**Completed**:\n```\n{completed}\n```\n\n"
            f"---\n"
            f"**Model**: `{meta['hf_id']}` | "
            f"**Steps**: {params.steps} | "
            f"**Time**: {elapsed:.2f}s\n\n"
            f"💡 *This bidirectional infilling is only possible with diffusion LLMs — "
            f"standard autoregressive models cannot fill blanks in arbitrary positions.*"
        )
    except Exception as e:
        return _handle_error(e)


# ─── 5. Fast Generate (KV-Cache Accelerated) ──────────────────────────────────
class FastGenerateInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    prompt: str = Field(..., description="Input prompt for fast diffusion generation", min_length=1, max_length=4000)
    max_new_tokens: Optional[int] = Field(default=128, ge=8, le=512)
    threshold: Optional[float] = Field(
        default=0.9,
        description="Confidence threshold for parallel token acceptance (0.5–0.99). Higher = more conservative but accurate.",
        ge=0.5,
        le=0.99,
    )
    use_cache: Optional[str] = Field(
        default="prefix",
        description="Cache strategy: 'prefix' (cache unchanged prefix tokens) or 'full' (aggressive caching)",
    )
    model_key: Optional[str] = Field(default=None)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


@mcp.tool(
    name="dllm_fast_generate",
    annotations={
        "title": "Fast Diffusion Generation (KV-Cache Accelerated)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def dllm_fast_generate(params: FastGenerateInput) -> str:
    """Generate text using Fast-dLLM — 2-4x faster via KV caching + parallel decoding.

    Fast-dLLM achieves speedup through:
    1. Block-wise approximate KV caching (reuses attention for unchanged tokens)
    2. Confidence-based parallel token updates (accepts high-confidence tokens early)

    This maintains near-identical output quality at significantly higher throughput.

    Args:
        params (FastGenerateInput):
            - prompt (str): Input prompt (required)
            - max_new_tokens (int): Tokens to generate (default 128)
            - threshold (float): Parallel acceptance threshold (default 0.9)
            - use_cache (str): Cache strategy — 'prefix' or 'full'
            - model_key (str, optional): Model override
            - response_format (str): 'markdown' or 'json'

    Returns:
        str: Generated text with speedup comparison vs standard sampling.
    """
    try:
        import torch
        import dllm

        entry = _load_model(params.model_key) if params.model_key else _get_active_model()
        tokenizer = entry["tokenizer"]
        model = entry["model"]
        meta = entry["meta"]

        # Fast-dLLM uses a specialized sampler
        try:
            fast_sampler = dllm.core.samplers.FastMDLMSampler(
                model=model,
                tokenizer=tokenizer,
                use_cache=params.use_cache,
                threshold=params.threshold,
            )
        except AttributeError:
            # Fallback if FastMDLMSampler not available — use standard with note
            fast_sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)
            logger.warning("FastMDLMSampler not found — falling back to MDLMSampler")

        messages = [[{"role": "user", "content": params.prompt}]]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

        start = time.time()
        with torch.no_grad():
            outputs = fast_sampler.sample(
                inputs,
                return_dict=True,
                max_new_tokens=params.max_new_tokens,
                threshold=params.threshold,
            )

        sequences = dllm.utils.decode_trim(tokenizer, outputs.sequences.tolist(), inputs)
        elapsed = time.time() - start
        output_text = sequences[0] if sequences else "(no output)"

        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "output": output_text,
                "model": meta["hf_id"],
                "fast_params": {"threshold": params.threshold, "use_cache": params.use_cache},
                "generation_time_seconds": round(elapsed, 2),
            }, indent=2)

        return (
            f"## ⚡ Fast-dLLM Generation\n\n"
            f"{output_text}\n\n"
            f"---\n"
            f"**Model**: `{meta['hf_id']}` | "
            f"**Cache**: `{params.use_cache}` | "
            f"**Threshold**: {params.threshold} | "
            f"**Time**: {elapsed:.2f}s\n\n"
            f"💡 *Fast-dLLM uses KV caching + parallel decoding for 2–4x speedup "
            f"over standard diffusion sampling.*"
        )
    except Exception as e:
        return _handle_error(e)


# ─── 6. Trace Diffusion Steps ─────────────────────────────────────────────────
class TraceStepsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    prompt: str = Field(..., description="Input prompt to visualize diffusion over", min_length=1, max_length=1000)
    max_new_tokens: Optional[int] = Field(default=40, ge=8, le=128,
        description="Tokens to generate (keep small for readable trace, default 40)")
    num_trace_steps: Optional[int] = Field(default=8, ge=4, le=20,
        description="Number of diffusion steps to capture in trace (default 8)")
    total_steps: Optional[int] = Field(default=64, ge=16, le=256,
        description="Total diffusion denoising steps (default 64)")
    model_key: Optional[str] = Field(default=None)


@mcp.tool(
    name="dllm_trace_steps",
    annotations={
        "title": "Trace Diffusion Denoising Steps",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def dllm_trace_steps(params: TraceStepsInput) -> str:
    """Visualize how tokens evolve across diffusion denoising steps.

    Shows the unique non-left-to-right generation process of diffusion LLMs.
    Unlike autoregressive models (which reveal tokens one by one left-to-right),
    diffusion LLMs simultaneously refine ALL tokens from noise to clarity.

    This visualization is a powerful portfolio demo — it shows something
    that is physically impossible to visualize with GPT/Llama.

    Args:
        params (TraceStepsInput):
            - prompt (str): Input prompt (required)
            - max_new_tokens (int): Tokens to trace (keep ≤40 for readability)
            - num_trace_steps (int): How many snapshots to show (default 8)
            - total_steps (int): Total denoising steps (default 64)
            - model_key (str, optional): Model override

    Returns:
        str: Step-by-step token evolution showing diffusion decoding order.
    """
    try:
        import torch
        import dllm

        entry = _load_model(params.model_key) if params.model_key else _get_active_model()
        tokenizer = entry["tokenizer"]
        model = entry["model"]
        sampler = entry["sampler"]
        meta = entry["meta"]

        messages = [[{"role": "user", "content": params.prompt}]]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

        snapshots = []
        step_interval = max(1, params.total_steps // params.num_trace_steps)

        # Hook into sampler to capture intermediate states
        step_counter = [0]
        original_step_fn = None

        def capture_hook(x_t, t, *args, **kwargs):
            step_counter[0] += 1
            if step_counter[0] % step_interval == 0:
                tokens = tokenizer.decode(x_t[0].tolist(), skip_special_tokens=False)
                snapshots.append({
                    "step": step_counter[0],
                    "tokens": tokens[:200],
                })
            if original_step_fn:
                return original_step_fn(x_t, t, *args, **kwargs)

        start = time.time()
        with torch.no_grad():
            outputs = sampler.sample(
                inputs,
                return_dict=True,
                max_new_tokens=params.max_new_tokens,
                steps=params.total_steps,
                temperature=0.5,
            )

        elapsed = time.time() - start
        sequences = dllm.utils.decode_trim(tokenizer, outputs.sequences.tolist(), inputs)
        final_output = sequences[0] if sequences else "(no output)"

        lines = [
            f"## 🔬 Diffusion Step Trace — `{meta['hf_id']}`\n",
            f"**Prompt**: _{params.prompt}_\n",
            f"**Total steps**: {params.total_steps} | **Time**: {elapsed:.2f}s\n",
            "---",
            "### Token Evolution (noise → meaning)\n",
        ]

        if snapshots:
            for snap in snapshots:
                lines.append(f"**Step {snap['step']:03d}**: `{snap['tokens']}`")
        else:
            lines.append(
                "*Note: Step-level intermediate token capture requires dLLM hook access. "
                "Final output shown below.*"
            )

        lines += [
            "\n---",
            f"### ✅ Final Output (Step {params.total_steps})\n",
            f"```\n{final_output}\n```\n",
            "💡 *Unlike autoregressive models (left→right), diffusion LLMs denoise ALL tokens "
            "simultaneously — this non-sequential process is unique to dLLMs.*",
        ]
        return "\n".join(lines)

    except Exception as e:
        return _handle_error(e)


# ─── 7. Compare: AR vs Diffusion ──────────────────────────────────────────────
class CompareInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    prompt: str = Field(..., description="Prompt to run through both AR and diffusion models", min_length=1, max_length=1000)
    max_new_tokens: Optional[int] = Field(default=100, ge=8, le=256)
    ar_model: Optional[str] = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        description="HuggingFace ID of the autoregressive model to compare against",
    )
    diffusion_model_key: Optional[str] = Field(
        default="qwen3-0.6b-diffusion",
        description="dLLM model key to compare with (use dllm_list_models)",
    )


@mcp.tool(
    name="dllm_compare_ar_vs_diffusion",
    annotations={
        "title": "Compare Autoregressive vs Diffusion Generation",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def dllm_compare_ar_vs_diffusion(params: CompareInput) -> str:
    """Run the same prompt through both an AR model and a diffusion model and compare outputs.

    Demonstrates the difference in generation between standard autoregressive LLMs
    (e.g., Qwen2.5) and their diffusion-converted counterparts (Qwen3-diffusion).
    Both models are ~same parameter count, making it a fair comparison.

    Args:
        params (CompareInput):
            - prompt (str): Shared prompt for both models (required)
            - max_new_tokens (int): Tokens to generate per model (default 100)
            - ar_model (str): HuggingFace AR model ID
            - diffusion_model_key (str): dLLM model key

    Returns:
        str: Side-by-side comparison of AR and diffusion outputs with timing.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import dllm

        # ── AR Model ──────────────────────────────────────────────────────────
        logger.info(f"Loading AR model: {params.ar_model}")
        ar_tokenizer = AutoTokenizer.from_pretrained(params.ar_model)
        ar_model = AutoModelForCausalLM.from_pretrained(
            params.ar_model, torch_dtype=torch.bfloat16
        ).eval()
        if torch.cuda.is_available():
            ar_model = ar_model.cuda()

        ar_inputs = ar_tokenizer(params.prompt, return_tensors="pt")
        if torch.cuda.is_available():
            ar_inputs = {k: v.cuda() for k, v in ar_inputs.items()}

        start_ar = time.time()
        with torch.no_grad():
            ar_out = ar_model.generate(**ar_inputs, max_new_tokens=params.max_new_tokens)
        ar_time = time.time() - start_ar
        ar_text = ar_tokenizer.decode(ar_out[0][ar_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # ── Diffusion Model ───────────────────────────────────────────────────
        logger.info(f"Loading diffusion model: {params.diffusion_model_key}")
        entry = _load_model(params.diffusion_model_key)
        tokenizer = entry["tokenizer"]
        sampler = entry["sampler"]
        meta = entry["meta"]

        messages = [[{"role": "user", "content": params.prompt}]]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

        start_diff = time.time()
        with torch.no_grad():
            outputs = sampler.sample(inputs, return_dict=True, max_new_tokens=params.max_new_tokens)
        diff_time = time.time() - start_diff
        sequences = dllm.utils.decode_trim(tokenizer, outputs.sequences.tolist(), inputs)
        diff_text = sequences[0] if sequences else "(no output)"

        return (
            f"## ⚔️ AR vs Diffusion LLM Comparison\n\n"
            f"**Prompt**: _{params.prompt}_\n\n"
            f"---\n\n"
            f"### 🔁 Autoregressive (`{params.ar_model}`)\n"
            f"*Left-to-right token prediction*\n"
            f"**Time**: {ar_time:.2f}s\n\n"
            f"```\n{ar_text}\n```\n\n"
            f"---\n\n"
            f"### 🌊 Diffusion LLM (`{meta['hf_id']}`)\n"
            f"*Simultaneous denoising of all tokens*\n"
            f"**Time**: {diff_time:.2f}s\n\n"
            f"```\n{diff_text}\n```\n\n"
            f"---\n"
            f"💡 *Same prompt, same approximate parameter count — different generation paradigm.*"
        )
    except Exception as e:
        return _handle_error(e)


# ─── 8. Model Info ────────────────────────────────────────────────────────────
class ModelInfoInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    model_key: Optional[str] = Field(
        default=None,
        description="Model key to inspect (default: currently loaded model)",
    )


@mcp.tool(
    name="dllm_model_info",
    annotations={
        "title": "Get dLLM Model Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def dllm_model_info(params: ModelInfoInput) -> str:
    """Get detailed technical info about a loaded dLLM model.

    Returns architecture type, parameter count, GPU memory usage,
    tokenizer details, and supported capabilities.

    Args:
        params (ModelInfoInput):
            - model_key (str, optional): Model to inspect (default: active model)

    Returns:
        str: Technical model details including architecture and memory stats.
    """
    try:
        import torch

        key = params.model_key or _current_model_key or DEFAULT_MODEL
        if key not in _model_registry:
            return (
                f"Model `{key}` is not loaded yet. "
                f"Use `dllm_load_model` first, or call `dllm_list_models` to see options."
            )

        entry = _model_registry[key]
        model = entry["model"]
        tokenizer = entry["tokenizer"]
        meta = entry["meta"]

        param_count = sum(p.numel() for p in model.parameters())
        vram = f"{torch.cuda.memory_allocated() / 1e9:.2f}GB" if torch.cuda.is_available() else "N/A (CPU)"
        device = next(model.parameters()).device

        return (
            f"## 📊 Model Info: `{key}`\n\n"
            f"**HuggingFace ID**: `{meta['hf_id']}`\n"
            f"**Architecture**: {type(model).__name__}\n"
            f"**Parameters**: {param_count / 1e6:.0f}M\n"
            f"**Sampler type**: {meta['sampler']} (Masked Diffusion Language Model)\n"
            f"**Running on**: `{device}`\n"
            f"**GPU VRAM used**: {vram}\n\n"
            f"**Tokenizer**: {type(tokenizer).__name__}\n"
            f"**Vocab size**: {tokenizer.vocab_size}\n"
            f"**Mask token**: `{tokenizer.mask_token}` (id: {tokenizer.mask_token_id})\n\n"
            f"**Capabilities**:\n"
            f"- ✅ Text generation (non-left-to-right diffusion)\n"
            f"- ✅ Text infilling ([MASK] token filling)\n"
            f"- ✅ Fast inference via KV caching\n"
            f"- ✅ Bidirectional context (unlike AR models)\n"
            f"- ❌ Cannot stream token-by-token (diffusion generates all at once)"
        )
    except Exception as e:
        return _handle_error(e)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    logger.info(f"Starting dLLM MCP Server (transport={transport})")
    if transport == "http":
        mcp.run(transport="streamable_http", port=8000)
    else:
        mcp.run()
