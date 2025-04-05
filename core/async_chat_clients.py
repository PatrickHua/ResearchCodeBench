import asyncio
import logging
import os
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI, RateLimitError

from core.data_classes.llm_type import LLMType, MODEL_CONFIGS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncChatClients():
    """
    LLM wrapper for async calls following OpenAI API format. Used for both OpenAI and other models.
    """
    def __init__(self) -> None:
        self.llm_clients = {
            
            'OPENAI': AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY')),
            'ANTHROPIC': AsyncOpenAI(api_key=os.getenv('ANTHROPIC_API_KEY'),
                                    base_url="https://api.anthropic.com/v1"),
            'GOOGLE': AsyncOpenAI(api_key=os.getenv('GOOGLE_API_KEY'),
                                  base_url="https://generativelanguage.googleapis.com/v1beta/"),
            'XAI': AsyncOpenAI(api_key=os.getenv('XAI_API_KEY'),
                               base_url="https://api.x.ai/v1"),
            'VLLM': AsyncOpenAI(api_key='EMPTY', base_url="http://localhost:8000/v1"),
            'DEEPSEEK': AsyncOpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'),
                                    base_url="https://api.deepseek.com/v1"),
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
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=timeout,
        ) | client_kwargs if client_kwargs else {}
        
        responses = []
        response_strs = []
        cost = 0
        for _ in range(num_completions):
            wait_time = 60
            response = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = await self.llm_clients[company].chat.completions.create(**full_kwargs)
                    break  # Success, exit retry loop
                except RateLimitError as e:
                    logger.warning(f"Rate limit exceeded for {company} on attempt {attempt}/{max_retries}: {e}. Retrying in {wait_time} seconds.")
                except Exception as e:
                    logger.error(f"Unexpected error for {company} on attempt {attempt}/{max_retries}: {e}. Retrying in {wait_time} seconds.")
                await asyncio.sleep(wait_time)
                wait_time *= 1

            if response is None:
                raise Exception(f"Failed to get a valid response from {company} after {max_retries} attempts.")

            responses.append(response)
            cost += self.calc_cost(response=response, llm_type=llm_type)
            response_strs.append(response.choices[0].message.content)

        full_response = {
            'response': response,
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
            # llm_type=LLMType.O1_HIGH,
            llm_type=LLMType.DEEPSEEK_R1,
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
