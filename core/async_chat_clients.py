import asyncio
import logging
import os
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI, RateLimitError
import httpx

from core.data_classes.llm_type import LLMType, MODEL_CONFIGS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a shared HTTP client with connection pooling
# This helps reuse connections across requests
http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=100,  # Increased from default
        max_keepalive_connections=20,
        keepalive_expiry=30.0  # Keep connections alive longer
    ),
    timeout=httpx.Timeout(60.0)  # Default timeout
)

class AsyncChatClients():
    """
    LLM wrapper for async calls following OpenAI API format. Used for both OpenAI and other models.
    """
    def __init__(self) -> None:
        # Use the shared HTTP client for all API clients
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

    def calc_cost(self, response, llm_type: LLMType) -> float:
        """
        Calculates the cost of a response.
        """
        llm_config = MODEL_CONFIGS[llm_type]
        input_cost = llm_config.input_cost / 1_000_000  # cost per 1M tokens
        output_cost = llm_config.output_cost / 1_000_000
        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        
        try:
            cost = (input_cost * input_tokens) + (output_cost * output_tokens)
        except KeyError:
            cost = 0

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        return cost

    # def get_prompt(
    #     self,
    #     llm_type: LLMType,
    #     user_message: str,
    #     system_message: Optional[str],
    # ) -> List[Dict[str, str]]:
    #     """
    #     Constructs the prompt for the model.
    #     """

    #     messages = [
    #         {"role": "system", "content": system_message},
    #         {"role": "user", "content": user_message},
    #     ]
    #     return messages

    # async def get_response(
    #     self,
    #     llm_type: LLMType,
    #     messages: List[Dict[str, str]],
    #     temperature: Optional[float] = None,
    #     max_tokens: Optional[int] = None,
    #     timeout: Optional[int] = None,
    #     max_retries: int = 10,
    # ) -> Any:
    #     """
    #     Attempts to get a valid response from the model using retries with exponential backoff.
    #     """
    #     company = MODEL_CONFIGS[llm_type].company
    #     model = llm_type.value

    #     wait_time = 60
    #     for attempt in range(1, max_retries + 1):
    #         try:
    #             return await self.llm_clients[company].chat.completions.create(
    #                     model=model,
    #                     messages=messages,
    #                     temperature=temperature,
    #                     max_completion_tokens=max_tokens,
    #                     timeout=timeout,
    #                     n=1
    #                 )

    #         except RateLimitError as e:
    #             logger.warning(f"Rate limit exceeded for {company} on attempt {attempt}/{max_retries}: {e}. Retrying in {wait_time} seconds.")
    #         except Exception as e:
    #             logger.error(f"Unexpected error for {company} on attempt {attempt}/{max_retries}: {e}. Retrying in {wait_time} seconds.")
    #         await asyncio.sleep(wait_time)
    #         wait_time *= 1

    #     raise Exception(f"Failed to get a valid response from {company} after {max_retries} attempts.")

    async def run(
        self,
        llm_type: LLMType,
        user_message: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        return_full_response: bool = False,
        num_completions: int = 1,
        max_retries: int = 10,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs the agent and returns the response along with cost information.
        """
        company = MODEL_CONFIGS[llm_type].company
        client_kwargs = MODEL_CONFIGS[llm_type].client_kwargs
        
        full_kwargs = dict(
            model=llm_type.value,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ] if system_message is not None else [
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=timeout,
            stream=stream,
        ) | client_kwargs

        # Instead of running sequentially, create tasks for all completions
        async def make_single_request():
            start_time = asyncio.get_event_loop().time()
            request_id = id(start_time)  # Simple unique ID for each request
            logger.info(f"[{request_id}] Starting request to {company} ({llm_type.value}) at {start_time:.3f}")
            
            wait_time = 60
            for attempt in range(1, max_retries + 1):
                try:
                    response = await self.llm_clients[company].chat.completions.create(**full_kwargs)
                    end_time = asyncio.get_event_loop().time()
                    duration = end_time - start_time
                    logger.info(f"[{request_id}] Completed request to {company} ({llm_type.value}) at {end_time:.3f} (took {duration:.3f}s)")
                    return response
                except RateLimitError as e:
                    logger.warning(f"[{request_id}] Rate limit exceeded for {company} on attempt {attempt}/{max_retries}: {e}. Retrying in {wait_time} seconds.")
                except Exception as e:
                    logger.error(f"[{request_id}] Unexpected error for {company} on attempt {attempt}/{max_retries}: {e}. Retrying in {wait_time} seconds.")
                await asyncio.sleep(wait_time)
                wait_time *= 1
            
            end_time = asyncio.get_event_loop().time()
            logger.error(f"[{request_id}] Failed request to {company} after {max_retries} attempts. Total duration: {end_time - start_time:.3f}s")
            raise Exception(f"Failed to get a valid response from {company} after {max_retries} attempts.")

        # Create tasks for all completions
        tasks = [make_single_request() for _ in range(num_completions)]
        
        # Run all tasks concurrently
        batch_start_time = asyncio.get_event_loop().time()
        logger.info(f"Starting batch of {num_completions} requests to {company} ({llm_type.value}) at {batch_start_time:.3f}")
        
        responses = await asyncio.gather(*tasks)
        
        batch_end_time = asyncio.get_event_loop().time()
        batch_duration = batch_end_time - batch_start_time
        logger.info(f"Completed batch of {num_completions} requests to {company} ({llm_type.value}) at {batch_end_time:.3f} (took {batch_duration:.3f}s)")
        
        response_strs = [response.choices[0].message.content for response in responses]
        cost = sum(self.calc_cost(response=response, llm_type=llm_type) for response in responses)

        full_response = {
            'response': responses[-1],  # Keep the last response for backward compatibility
            'response_str': response_strs,
            'cost': cost
        }
        self.total_inference_cost += cost
        self.all_responses.append(full_response)
        if return_full_response:
            return full_response
        else:
            return response_strs

    async def batch_prompt(
        self,
        user_messages: List[str],
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        llm_type: Optional[LLMType] = None,
    ) -> List[str]:
        responses = [
            self.run(
                llm_type=llm_type,
                user_message=msg,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_tokens
            ) for msg in user_messages
        ]
        return await asyncio.gather(*responses)


async def main():
    clients = AsyncChatClients()
    try:
        output = await clients.run(
            # llm_type=LLMType.GEMINI_2_0_FLASH_LITE_PREVIEW_02_05,
            # llm_type=LLMType.O3_MINI_2025_01_31,
            # llm_type=LLMType.CLAUDE_3_5_SONNET_2024_10_22,
            # llm_type=LLMType.O3_MINI_HIGH,
            # llm_type=LLMType.CLAUDE_3_7_SONNET_2025_02_19,
            llm_type=LLMType.O1_HIGH,
            # llm_type=LLMType.DEEPSEEK_R1,
            # llm_type=LLMType.GEMINI_2_0_FLASH,
            user_message='Hello, how are you?',
            system_message='You are a helpful assistant.',
            num_completions=2,
            temperature=0,
            # max_tokens=4096,
        )
        print("Response:", output)
        breakpoint()
    except Exception as e:
        logger.error(f"Error during main execution: {e}")

if __name__ == '__main__':
    asyncio.run(main())
