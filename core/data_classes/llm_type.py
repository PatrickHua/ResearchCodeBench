from enum import Enum
from typing import Optional
import os
from pydantic import BaseModel

class LLMType(str, Enum):
    
    # Open source models
    QWEN_2_5_1_5B_INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"
    DEEPSEEK_R1_DISTILL_QWEN_14B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    QWEN_2_5_72B_INSTRUCT = "Qwen/Qwen2.5-72B-Instruct"
    
    # OpenAI models
    GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"
    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"  #  = GPT_4O_2024_08_06
    GPT_4O_2024_11_20 = "gpt-4o-2024-11-20"
    GPT_4O_2024_08_06 = "gpt-4o-2024-08-06"
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    # O1_MINI = "o1-mini"
    O1="o1-2024-12-17"
    O1_HIGH="o1-2024-12-17"
    O1_PREVIEW_2024_09_12 = "o1-preview-2024-09-12"
    # O3_MINI_2025_01_31_HIGH = "o3-mini-2025-01-31-high"
    O3_MINI_HIGH = "o3-mini-2025-01-31"
    O3_MINI_2025_01_31 = "o3-mini-2025-01-31"
    O3_MINI = "o3-mini-2025-01-31"  # same with o3-mini-2025-01-31
    # O1 = "o1"  # not working
    O1_MINI = "o1-mini-2024-09-12"
    # O1_MINI_HIGH = "o1-mini-2024-09-12"
    
    # Anthropic models https://docs.anthropic.com/en/docs/about-claude/models
    CLAUDE_3_7_SONNET_2025_02_19 = "claude-3-7-sonnet-20250219"
    CLAUDE_3_5_SONNET_2024_10_22 = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU_2024_10_22 = "claude-3-5-haiku-20241022"
    CLAUDE_3_OPUS_2024_02_29 = "claude-3-opus-20240229"
    CLAUDE_3_SONNET_2024_02_29 = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU_2024_03_07 = "claude-3-haiku-20240307"
    
    # Gemini models  https://ai.google.dev/gemini-api/docs/models/gemini
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_FLASH_8B = "gemini-1.5-flash-8b"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_FLASH_LITE_PREVIEW_02_05 = "gemini-2.0-flash-lite-preview-02-05"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    # GEMINI_1_0_PRO = "gemini-1.0-pro"  # Deprecated on 2/15/2025
    
    # XAI models
    GROK_2_1212 = "grok-2-1212"

    # DeepSeek models
    DEEPSEEK_R1 = "deepseek-reasoner"


class LLMConfig(BaseModel):
    company: str
    input_cost: Optional[float]
    output_cost: Optional[float]
    knowledge_cutoff_date: Optional[str] = None
    client_kwargs: dict = {}


# Define configurations for each model using the enum values
MODEL_CONFIGS = {
    LLMType.QWEN_2_5_1_5B_INSTRUCT: LLMConfig(company='VLLM', input_cost=0, output_cost=0),
    LLMType.DEEPSEEK_R1_DISTILL_QWEN_14B: LLMConfig(company='VLLM', input_cost=0, output_cost=0),
    LLMType.QWEN_2_5_72B_INSTRUCT: LLMConfig(company='VLLM', input_cost=0, output_cost=0),
    
    # https://platform.openai.com/docs/pricing
    LLMType.GPT_3_5_TURBO_0125: LLMConfig(company='OPENAI', input_cost=0.5, output_cost=1.5),
    LLMType.GPT_4_TURBO_2024_04_09: LLMConfig(company='OPENAI', input_cost=10, output_cost=30),
    LLMType.GPT_4O_MINI: LLMConfig(company='OPENAI', input_cost=0.15, output_cost=0.6),
    LLMType.GPT_4O: LLMConfig(company='OPENAI', input_cost=2.5, output_cost=10),
    LLMType.GPT_4O_2024_11_20: LLMConfig(company='OPENAI', input_cost=2.5, output_cost=10),
    LLMType.GPT_4O_2024_08_06: LLMConfig(company='OPENAI', input_cost=2.5, output_cost=10),
    LLMType.GPT_4O_2024_05_13: LLMConfig(company='OPENAI', input_cost=5, output_cost=15),
    LLMType.O1_MINI: LLMConfig(company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1}),
    # LLMType.O3_MINI_2025_01_31_HIGH: LLMConfig(company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1, 'reasoning': {"effort": "high"}}),


    LLMType.O3_MINI_HIGH: LLMConfig(company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1, 'reasoning': {"effort": "high"}}),
    LLMType.O3_MINI_2025_01_31: LLMConfig(company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1}),
    LLMType.O3_MINI: LLMConfig(company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1}),
    LLMType.O1_PREVIEW_2024_09_12: LLMConfig(company='OPENAI', input_cost=15, output_cost=60, client_kwargs={'temperature': 1}),
    LLMType.O1: LLMConfig(company='OPENAI', input_cost=15, output_cost=60, client_kwargs={'temperature': 1}),
    LLMType.O1_HIGH: LLMConfig(company='OPENAI', input_cost=15, output_cost=60, client_kwargs={'temperature': 1, 'reasoning': {"effort": "high"}}),
    LLMType.O1_MINI: LLMConfig(company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1}),
    # LLMType.O1_MINI_HIGH: LLMConfig(company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1, 'reasoning': {"effort": "high"}}),
    
    # Anthropic models https://www.anthropic.com/pricing#anthropic-api
    # https://docs.anthropic.com/en/docs/about-claude/models
    LLMType.CLAUDE_3_7_SONNET_2025_02_19: LLMConfig(company='ANTHROPIC', input_cost=3, output_cost=15, client_kwargs={'max_completion_tokens': 8192}),
    LLMType.CLAUDE_3_5_SONNET_2024_10_22: LLMConfig(company='ANTHROPIC', input_cost=3, output_cost=15, client_kwargs={'max_completion_tokens': 8192}),
    LLMType.CLAUDE_3_5_HAIKU_2024_10_22: LLMConfig(company='ANTHROPIC', input_cost=0.8, output_cost=4, client_kwargs={'max_completion_tokens': 8192}),
    LLMType.CLAUDE_3_OPUS_2024_02_29: LLMConfig(company='ANTHROPIC', input_cost=15, output_cost=75, client_kwargs={'max_completion_tokens': 4096}),
    LLMType.CLAUDE_3_SONNET_2024_02_29: LLMConfig(company='ANTHROPIC', input_cost=3, output_cost=15, client_kwargs={'max_completion_tokens': 4096}),
    LLMType.CLAUDE_3_HAIKU_2024_03_07: LLMConfig(company='ANTHROPIC', input_cost=0.25, output_cost=1.25, client_kwargs={'max_completion_tokens': 4096}),

    # https://ai.google.dev/pricing#2_0flash
    LLMType.GEMINI_2_0_FLASH: LLMConfig(company='GOOGLE', input_cost=0.1, output_cost=0.4),
    LLMType.GEMINI_2_0_FLASH_LITE_PREVIEW_02_05: LLMConfig(company='GOOGLE', input_cost=0.075, output_cost=0.30),
    LLMType.GEMINI_1_5_FLASH: LLMConfig(company='GOOGLE', input_cost=0.075, output_cost=0.3),
    LLMType.GEMINI_1_5_FLASH_8B: LLMConfig(company='GOOGLE', input_cost=0.0375, output_cost=0.15),
    LLMType.GEMINI_1_5_PRO: LLMConfig(company='GOOGLE', input_cost=1.25, output_cost=5.00),
    # LLMType.GEMINI_1_0_PRO: LLMConfig(company='GOOGLE', input_cost=0.50, output_cost=1.50),

    LLMType.DEEPSEEK_R1: LLMConfig(company='DEEPSEEK', input_cost=0.14, output_cost=2.19, client_kwargs={'stream': False}),


    # https://x.ai/pricing
    LLMType.GROK_2_1212: LLMConfig(company='XAI', input_cost=2, output_cost=10, knowledge_cutoff_date="2024-07-17"),
}


if __name__ == "__main__":
    print(MODEL_CONFIGS[LLMType.GPT_4O])
    print(LLMType.GPT_4O)
    print(str(LLMType.GPT_4O))
    print(f"{LLMType.GPT_4O}")
    print(f"{LLMType.GPT_4O.value}")
    print(f"{LLMType.GPT_4O.name}")
 