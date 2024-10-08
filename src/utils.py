import os
import logging
from http import HTTPStatus
from functools import wraps
from time import time
import requests
import tarfile

try:
    from vllm.utils import random_uuid
    from vllm.entrypoints.openai.protocol import ErrorResponse
    from vllm import SamplingParams
    from vllm.lora import LoRARequest
except ImportError:
    logging.warning("Error importing vllm, skipping related imports. This is ONLY expected when baking model into docker image from a machine without GPUs")
    pass

logging.basicConfig(level=logging.INFO)

def count_physical_cores():
    with open('/proc/cpuinfo') as f:
        content = f.readlines()

    cores = set()
    current_physical_id = None
    current_core_id = None

    for line in content:
        if 'physical id' in line:
            current_physical_id = line.strip().split(': ')[1]
        elif 'core id' in line:
            current_core_id = line.strip().split(': ')[1]
            cores.add((current_physical_id, current_core_id))

    return len(cores)

def get_top_level_directory(tar_path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        first_member = tar.next()
        if first_member.isdir():
            return first_member.name
    return None

def extract_tarfile(file_path, extract_path):
    top_level_dir = get_top_level_directory(file_path)
    
    with tarfile.open(file_path, 'r:gz') as tar:
        if top_level_dir and top_level_dir == os.path.basename(extract_path):
            # Extract to parent directory to avoid creating a nested directory
            tar.extractall(path=os.path.dirname(extract_path))
        else:
            tar.extractall(path=extract_path)

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    logging.info(f"Downloaded {url} to {local_filename}")
    return local_filename

class JobInput:
    def __init__(self, job):
        self.llm_input = job.get("messages", job.get("prompt"))
        self.stream = job.get("stream", False)
        self.max_batch_size = job.get("max_batch_size")
        self.apply_chat_template = job.get("apply_chat_template", False)
        self.use_openai_format = job.get("use_openai_format", False)
        self.sampling_params = SamplingParams(**job.get("sampling_params", {}))
        self.request_id = random_uuid()
        batch_size_growth_factor = job.get("batch_size_growth_factor")
        self.batch_size_growth_factor = float(batch_size_growth_factor) if batch_size_growth_factor else None 
        min_batch_size = job.get("min_batch_size")
        self.min_batch_size = int(min_batch_size) if min_batch_size else None 
        self.openai_route = job.get("openai_route")
        self.openai_input = job.get("openai_input")
        self.lora_request = LoRARequest("safecoder", 1, "/tmp/mistral-7b-lora-safecoder")

class DummyRequest:
    async def is_disconnected(self):
        return False
    
class BatchSize:
    def __init__(self, max_batch_size, min_batch_size, batch_size_growth_factor):
        self.max_batch_size = max_batch_size
        self.batch_size_growth_factor = batch_size_growth_factor
        self.min_batch_size = min_batch_size
        self.is_dynamic = batch_size_growth_factor > 1 and min_batch_size >= 1 and max_batch_size > min_batch_size
        if self.is_dynamic:
            self.current_batch_size = min_batch_size
        else:
            self.current_batch_size = max_batch_size
        
    def update(self):
        if self.is_dynamic:
            self.current_batch_size = min(self.current_batch_size*self.batch_size_growth_factor, self.max_batch_size)
        
def create_error_response(message: str, err_type: str = "BadRequestError", status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
    return ErrorResponse(message=message,
                            type=err_type,
                            code=status_code.value)
    
def get_int_bool_env(env_var: str, default: bool) -> bool:
    return int(os.getenv(env_var, int(default))) == 1

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        logging.info(f"{func.__name__} completed in {end - start:.2f} seconds")
        return result
    return wrapper