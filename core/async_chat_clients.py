import asyncio
import logging
import os
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI, RateLimitError
import httpx

from core.data_classes.llm_type import LLMType, MODEL_CONFIGS
import asyncio
import logging
import random
import os
from typing import Any, Optional

from openai import AsyncOpenAI, RateLimitError
import httpx

# Assuming MODEL_CONFIGS and LLMType are already defined...

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared HTTP client (as in your code)
http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=30.0
    ),
    timeout=httpx.Timeout(60.0)
)

class AsyncChatClients:
    def __init__(self) -> None:
        self.llm_clients = {
            'OPENAI': AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'), http_client=http_client),
            'ANTHROPIC': AsyncOpenAI(api_key=os.getenv('ANTHROPIC_API_KEY'),
                                      base_url="https://api.anthropic.com/v1",
                                      http_client=http_client),
            'GOOGLE': AsyncOpenAI(api_key=os.getenv('GOOGLE_API_KEY'),
                                  base_url="https://generativelanguage.googleapis.com/v1beta/",
                                  http_client=http_client),
            'XAI': AsyncOpenAI(api_key=os.getenv('XAI_API_KEY'),
                               base_url="https://api.x.ai/v1",
                               http_client=http_client),
            'VLLM': AsyncOpenAI(api_key='EMPTY', base_url="http://localhost:8000/v1",
                               http_client=http_client),
            'DEEPSEEK': AsyncOpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'),
                                    base_url="https://api.deepseek.com/v1",
                                    http_client=http_client),
        }
        self.all_responses = []
        self.total_inference_cost = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def calc_cost(self, response, llm_type) -> float:
        llm_config = MODEL_CONFIGS[llm_type]
        input_cost = llm_config.input_cost / 1_000_000  # cost per 1M tokens
        output_cost = llm_config.output_cost / 1_000_000

        
        if llm_type == LLMType.O1_HIGH:
            output_tokens = response.usage.output_tokens
            input_tokens = response.usage.input_tokens
        else:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
        cost = (input_cost * input_tokens) + (output_cost * output_tokens)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        return cost

    async def run(
        self,
        llm_type,
        user_message: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        return_full_response: bool = False,
        num_completions: int = 1,
        max_retries: int = 10,
        stream: bool = False,
    ) -> Any:
        company = MODEL_CONFIGS[llm_type].company
        client_kwargs = MODEL_CONFIGS[llm_type].client_kwargs

        full_kwargs = dict(
            model=llm_type.value,
            messages=(
                [{"role": "system", "content": system_message},
                 {"role": "user", "content": user_message}]
                if system_message is not None
                else [{"role": "user", "content": user_message}]
            ),
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=timeout,
            stream=stream,
        ) | client_kwargs

        async def make_single_request():
            start_time = asyncio.get_event_loop().time()
            request_id = id(start_time)
            logger.info(f"[{request_id}] Starting request to {company} ({llm_type.value}) at {start_time:.3f}")
            
            # Start with a modest wait time
            wait_time = 1  
            for attempt in range(1, max_retries + 1):
                try:
                    if llm_type == LLMType.O1_HIGH:
                        # For O1, we need to use input instead of messages
                        o1_kwargs = full_kwargs.copy()
                        o1_kwargs["input"] = o1_kwargs.pop("messages")
                        if "max_completion_tokens" in o1_kwargs:
                            o1_kwargs["max_output_tokens"] = o1_kwargs.pop("max_completion_tokens")
                        response = await self.llm_clients[company].responses.create(**o1_kwargs)
                    else:
                        response = await self.llm_clients[company].chat.completions.create(**full_kwargs)
                    end_time = asyncio.get_event_loop().time()
                    duration = end_time - start_time
                    logger.info(f"[{request_id}] Completed request to {company} ({llm_type.value}) at {end_time:.3f} (took {duration:.3f}s)")
                    return response
                except RateLimitError as e:
                    # Check if the API provided a "Retry-After" header
                    retry_after = None
                    if hasattr(e, 'response') and e.response is not None:
                        retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            # Use the value from the header as the wait time
                            wait_time = float(retry_after)
                            logger.warning(f"[{request_id}] Rate limit exceeded on attempt {attempt}/{max_retries}: {str(e)}. 'Retry-After' header found, retrying in {wait_time:.2f}s")
                        except ValueError:
                            logger.warning(f"[{request_id}] Rate limit exceeded on attempt {attempt}/{max_retries}: {str(e)}. Invalid 'Retry-After' header value, using default wait time {wait_time:.2f}s")
                    else:
                        logger.warning(f"[{request_id}] Rate limit exceeded on attempt {attempt}/{max_retries}: {str(e)}. No 'Retry-After' header found, retrying in {wait_time:.2f}s")
                except Exception as e:
                    logger.error(f"[{request_id}] Unexpected error on attempt {attempt}/{max_retries}: {str(e)}, type: {type(e).__name__}. Full error: {repr(e)}. Retrying in {wait_time:.2f}s")
                    if hasattr(e, 'response'):
                        logger.error(f"Response details: {e.response.text if hasattr(e.response, 'text') else e.response}")
                
                # Add jitter to avoid thundering herd effect
                jitter = random.uniform(0, wait_time)
                sleep_time = wait_time + jitter
                await asyncio.sleep(sleep_time)
                # Exponential backoff: double the wait time for next attempt
                wait_time *= 2
            
            end_time = asyncio.get_event_loop().time()
            logger.error(f"[{request_id}] Failed request to {company} after {max_retries} attempts (total duration: {end_time - start_time:.3f}s)")
            raise Exception(f"Failed to get a valid response from {company} after {max_retries} attempts.")

        tasks = [make_single_request() for _ in range(num_completions)]
        batch_start_time = asyncio.get_event_loop().time()
        logger.info(f"Starting batch of {num_completions} requests to {company} ({llm_type.value}) at {batch_start_time:.3f}")
        
        responses = await asyncio.gather(*tasks)
        
        batch_end_time = asyncio.get_event_loop().time()
        batch_duration = batch_end_time - batch_start_time
        logger.info(f"Completed batch of {num_completions} requests to {company} ({llm_type.value}) at {batch_end_time:.3f} (took {batch_duration:.3f}s)")
        
        # Handle different response formats
        if llm_type == LLMType.O1_HIGH:
            response_strs = [response.output_text for response in responses]
        else:
            response_strs = [response.choices[0].message.content for response in responses]

        cost = sum(self.calc_cost(response=response, llm_type=llm_type) for response in responses)

        full_response = {
            'response': responses[-1],
            'response_str': response_strs,
            'cost': cost
        }
        self.total_inference_cost += cost
        self.all_responses.append(full_response)
        return full_response if return_full_response else response_strs

# Example usage:
async def main():
    clients = AsyncChatClients()
    try:
        output = await clients.run(
            llm_type=LLMType.O1_HIGH,  # Adjust to your model type
            user_message='Hello, how are you?',
            system_message='You are a helpful assistant.',
            num_completions=2,
            temperature=0,
        )
        print("Response:", output)
    except Exception as e:
        logger.error(f"Error during main execution: {e}")

if __name__ == '__main__':
    asyncio.run(main())
