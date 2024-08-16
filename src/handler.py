import os
import runpod
from utils import JobInput, download_file, extract_tarfile
from engine import vLLMEngine, OpenAIvLLMEngine

vllm_engine = vLLMEngine()
OpenAIvLLMEngine = OpenAIvLLMEngine(vllm_engine)

# For now, we setup the LoRA for SafeCoder here
# This will be generalized in the future

# Download the Mistral LoRA module
mistral_lora_url = "https://files.sri.inf.ethz.ch/safecoder/mistral-7b-lora-safecoder.tar.gz"
mistral_lora_path = "/tmp/mistral-7b-lora-safecoder.tar.gz"
extract_path = "/tmp/mistral-7b-lora-safecoder"

download_file(mistral_lora_url, mistral_lora_path)

# Extract
os.makedirs(extract_path, exist_ok=True)
extract_tarfile(mistral_lora_path, extract_path)
print(f"File extracted to: {extract_path}")

# Optionally, remove the downloaded .tar.gz file
os.remove(mistral_lora_path)
print(f"Removed downloaded file: {mistral_lora_path}")


async def handler(job):
    job_input = JobInput(job["input"])
    engine = OpenAIvLLMEngine if job_input.openai_route else vllm_engine
    results_generator = engine.generate(job_input)
    async for batch in results_generator:
        yield batch

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)