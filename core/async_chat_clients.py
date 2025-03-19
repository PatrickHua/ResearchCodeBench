import asyncio
import logging
import os
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError
import openai
from anthropic import AsyncAnthropic, RateLimitError as AnthropicRateLimitError, InternalServerError

from core.data_classes.llm_type import LLMType, MODEL_CONFIGS

# Companies using specific clients
COMPANIES_USING_OPENAI_CLIENT = ['OPENAI', 'VLLM', 'GOOGLE', 'XAI']
COMPANIES_USING_ANTHROPIC_CLIENT = ['ANTHROPIC']

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
            'ANTHROPIC': AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY')),
            'GOOGLE': AsyncOpenAI(api_key=os.getenv('GOOGLE_API_KEY'),
                                  base_url="https://generativelanguage.googleapis.com/v1beta/"),
            'XAI': AsyncOpenAI(api_key=os.getenv('XAI_API_KEY'),
                               base_url="https://api.x.ai/v1"),
            'VLLM': AsyncOpenAI(api_key='EMPTY', base_url="http://localhost:8000/v1"),
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
        company = llm_config.company
        input_cost = llm_config.input_cost / 1_000_000  # cost per 1M tokens
        output_cost = llm_config.output_cost / 1_000_000
        
        if company in COMPANIES_USING_OPENAI_CLIENT:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
        elif company in COMPANIES_USING_ANTHROPIC_CLIENT:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        else:
            raise ValueError(f"Unknown company: {company}")
        
        try:
            cost = (input_cost * input_tokens) + (output_cost * output_tokens)
        except KeyError:
            cost = 0

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        return cost

    def get_prompt(
        self,
        llm_type: LLMType,
        user_message: str,
        system_message: Optional[str],
    ) -> List[Dict[str, str]]:
        """
        Constructs the prompt for the model.
        """
        model = llm_type.value
        company = MODEL_CONFIGS[llm_type].company

        if company == 'ANTHROPIC':
            messages = [
                system_message if system_message else [],
                {"role": "user", "content": user_message},
            ]
        elif system_message is None or model in ['mistralai/Mixtral-8x7B-Instruct-v0.1']:
            messages = [{"role": "user", "content": user_message}]
        else:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        return messages

    async def get_response(
        self,
        llm_type: LLMType,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: int = 10,
        num_completions: int = 1,
    ) -> Any:
        """
        Attempts to get a valid response from the model using retries with exponential backoff.
        """
        company = MODEL_CONFIGS[llm_type].company
        model = llm_type.value

        wait_time = 60
        for attempt in range(1, max_retries + 1):
            try:
                if company in COMPANIES_USING_OPENAI_CLIENT:
                    return await self.llm_clients[company].chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        timeout=timeout,
                        n=num_completions
                    )
                elif company in COMPANIES_USING_ANTHROPIC_CLIENT:
                    system_message, remaining_messages = messages[0], messages[1:]
                    return await self.llm_clients[company].messages.create(
                        model=model,
                        messages=remaining_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system=system_message,
                        timeout=timeout,
                        n=num_completions
                    )
                else:
                    raise ValueError(f"Unknown company: {company}")
            except (OpenAIRateLimitError, AnthropicRateLimitError) as e:
                logger.warning(f"Rate limit exceeded for {company} on attempt {attempt}/{max_retries}: {e}. Retrying in {wait_time} seconds.")
            except InternalServerError as e:
                logger.warning(f"Internal server error for {company} on attempt {attempt}/{max_retries}: {e}. Retrying in {wait_time} seconds.")
            except Exception as e:
                logger.error(f"Unexpected error for {company} on attempt {attempt}/{max_retries}: {e}. Retrying in {wait_time} seconds.")
            await asyncio.sleep(wait_time)
            wait_time *= 1

        raise Exception(f"Failed to get a valid response from {company} after {max_retries} attempts.")

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
    ) -> Dict[str, Any]:
        """
        Runs the agent and returns the response along with cost information.
        """
        company = MODEL_CONFIGS[llm_type].company

        if company in COMPANIES_USING_ANTHROPIC_CLIENT:
            assert temperature is not None, 'temperature must be provided for Anthropic models'
            assert max_tokens is not None, 'max_tokens must be provided for Anthropic models'
        elif company == 'OPENAI' and llm_type.value.startswith('o'):
            temperature = 1
            max_tokens = max_tokens * 10 if max_tokens is not None else None

        model = llm_type.value
        messages = self.get_prompt(llm_type=llm_type, user_message=user_message, system_message=system_message)
        
        response = await self.get_response(
            llm_type=llm_type,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            num_completions=num_completions
        )

        # Validate response
        if hasattr(response, 'choices') and response.choices is None:
            logger.error(f"Invalid response received for model {model}: {response}")
            raise ValueError(f"Invalid response received for model {model}")

        # Calculate cost and parse response string
        cost = self.calc_cost(response=response, llm_type=llm_type)
        if company in COMPANIES_USING_OPENAI_CLIENT:
            response_str = [choice.message.content for choice in response.choices]
        elif company in COMPANIES_USING_ANTHROPIC_CLIENT:
            response_str = [content.text for content in response.content]
        else:
            raise ValueError(f"Unknown company: {company}")

        full_response = {
            'response': response,
            'response_str': response_str,
            'cost': cost
        }
        self.total_inference_cost += cost
        self.all_responses.append(full_response)
        if return_full_response:
            return full_response
        else:
            return response_str

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
            llm_type=LLMType.GEMINI_2_0_FLASH_LITE_PREVIEW_02_05,
            user_message='Hello, how are you? What is lsikjdfgolw;ajg;o',
            system_message='You are a helpful assistant.',
            num_completions=2,
            temperature=0.5
        )
        print("Response:", output)
        breakpoint()
    except Exception as e:
        logger.error(f"Error during main execution: {e}")

if __name__ == '__main__':
    asyncio.run(main())
