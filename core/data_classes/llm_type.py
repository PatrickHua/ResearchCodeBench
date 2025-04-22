from enum import Enum, auto
from typing import Optional
import os
from pydantic import BaseModel

class LLMType(str, Enum):
    
    # Open source models
    QWEN_2_5_1_5B_INSTRUCT = auto()
    DEEPSEEK_R1_DISTILL_QWEN_14B = auto()
    QWEN_2_5_72B_INSTRUCT = auto()
    
    # OpenAI models
    GPT_3_5_TURBO_0125 = auto()
    GPT_4_TURBO_2024_04_09 = auto()
    GPT_4O_MINI = auto()
    GPT_4O = auto()  #  = GPT_4O_2024_08_06
    GPT_4O_2024_11_20 = auto()
    GPT_4O_2024_08_06 = auto()
    GPT_4O_2024_05_13 = auto()
    # O1_MINI = "o1-mini"
    O1 = auto()
    O1_HIGH = auto()
    O1_PREVIEW_2024_09_12 = auto()
    # O3_MINI_2025_01_31_HIGH = "o3-mini-2025-01-31-high"
    O3_MINI_HIGH = auto()
    O3_MINI_2025_01_31 = auto()
    O3_MINI = auto()  # same with o3-mini-2025-01-31
    # O1 = "o1"  # not working
    O1_MINI = auto()
    # O1_MINI_HIGH = "o1-mini-2024-09-12"
    
    # Anthropic models https://docs.anthropic.com/en/docs/about-claude/models
    CLAUDE_3_7_SONNET_2025_02_19 = auto()
    CLAUDE_3_5_SONNET_2024_10_22 = auto()
    CLAUDE_3_5_HAIKU_2024_10_22 = auto()
    CLAUDE_3_OPUS_2024_02_29 = auto()
    CLAUDE_3_SONNET_2024_02_29 = auto()
    CLAUDE_3_HAIKU_2024_03_07 = auto()
    
    # Gemini models  https://ai.google.dev/gemini-api/docs/models/gemini
    GEMINI_1_5_FLASH = auto()
    GEMINI_1_5_FLASH_8B = auto()
    GEMINI_2_0_FLASH = auto()
    GEMINI_2_0_FLASH_LITE_PREVIEW_02_05 = auto()
    GEMINI_1_5_PRO = auto()
    # GEMINI_1_0_PRO = "gemini-1.0-pro"  # Deprecated on 2/15/2025
    
    # XAI models
    GROK_2_1212 = auto()

    # DeepSeek models
    DEEPSEEK_R1 = auto()


class LLMConfig(BaseModel):
    model: str
    company: str
    input_cost: Optional[float]
    output_cost: Optional[float]
    knowledge_cutoff_date: Optional[str] = None
    client_kwargs: dict = {}


# Define configurations for each model using the enum values
MODEL_CONFIGS = {
    LLMType.QWEN_2_5_1_5B_INSTRUCT: LLMConfig(model="Qwen/Qwen2.5-1.5B-Instruct", company='VLLM', input_cost=0, output_cost=0),
    LLMType.DEEPSEEK_R1_DISTILL_QWEN_14B: LLMConfig(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", company='VLLM', input_cost=0, output_cost=0),
    LLMType.QWEN_2_5_72B_INSTRUCT: LLMConfig(model="Qwen/Qwen2.5-72B-Instruct", company='VLLM', input_cost=0, output_cost=0),
    
    # https://platform.openai.com/docs/pricing
    LLMType.GPT_3_5_TURBO_0125: LLMConfig(model="gpt-3.5-turbo-0125", company='OPENAI', input_cost=0.5, output_cost=1.5),
    LLMType.GPT_4_TURBO_2024_04_09: LLMConfig(model="gpt-4-turbo-2024-04-09", company='OPENAI', input_cost=10, output_cost=30),
    LLMType.GPT_4O_MINI: LLMConfig(model="gpt-4o-mini", company='OPENAI', input_cost=0.15, output_cost=0.6),
    LLMType.GPT_4O: LLMConfig(model="gpt-4o", company='OPENAI', input_cost=2.5, output_cost=10),
    LLMType.GPT_4O_2024_11_20: LLMConfig(model="gpt-4o-2024-11-20", company='OPENAI', input_cost=2.5, output_cost=10),
    LLMType.GPT_4O_2024_08_06: LLMConfig(model="gpt-4o-2024-08-06", company='OPENAI', input_cost=2.5, output_cost=10),
    LLMType.GPT_4O_2024_05_13: LLMConfig(model="gpt-4o-2024-05-13", company='OPENAI', input_cost=5, output_cost=15),
    LLMType.O1_MINI: LLMConfig(model="o1-mini-2024-09-12", company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1}),
    # LLMType.O3_MINI_2025_01_31_HIGH: LLMConfig(company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1, 'reasoning': {"effort": "high"}}),


    LLMType.O3_MINI_HIGH: LLMConfig(model="o3-mini-2025-01-31", company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1, 'reasoning': {"effort": "high"}}),
    LLMType.O3_MINI_2025_01_31: LLMConfig(model="o3-mini-2025-01-31", company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1}),
    LLMType.O3_MINI: LLMConfig(model="o3-mini-2025-01-31", company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1}),
    LLMType.O1_PREVIEW_2024_09_12: LLMConfig(model="o1-preview-2024-09-12", company='OPENAI', input_cost=15, output_cost=60, client_kwargs={'temperature': 1}),
    LLMType.O1: LLMConfig(model="o1-2024-12-17", company='OPENAI', input_cost=15, output_cost=60, client_kwargs={'temperature': 1}),
    LLMType.O1_HIGH: LLMConfig(model="o1-2024-12-17", company='OPENAI', input_cost=15, output_cost=60, client_kwargs={'temperature': 1, 'reasoning': {"effort": "high"}}),
    LLMType.O1_MINI: LLMConfig(model="o1-mini-2024-09-12", company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1}),
    # LLMType.O1_MINI_HIGH: LLMConfig(company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1, 'reasoning': {"effort": "high"}}),
    
    # Anthropic models https://www.anthropic.com/pricing#anthropic-api
    # https://docs.anthropic.com/en/docs/about-claude/models
    LLMType.CLAUDE_3_7_SONNET_2025_02_19: LLMConfig(model="claude-3-7-sonnet-20250219", company='ANTHROPIC', input_cost=3, output_cost=15, client_kwargs={'max_completion_tokens': 8192}),
    LLMType.CLAUDE_3_5_SONNET_2024_10_22: LLMConfig(model="claude-3-5-sonnet-20241022", company='ANTHROPIC', input_cost=3, output_cost=15, client_kwargs={'max_completion_tokens': 8192}),
    LLMType.CLAUDE_3_5_HAIKU_2024_10_22: LLMConfig(model="claude-3-5-haiku-20241022", company='ANTHROPIC', input_cost=0.8, output_cost=4, client_kwargs={'max_completion_tokens': 8192}),
    LLMType.CLAUDE_3_OPUS_2024_02_29: LLMConfig(model="claude-3-opus-20240229", company='ANTHROPIC', input_cost=15, output_cost=75, client_kwargs={'max_completion_tokens': 4096}),
    LLMType.CLAUDE_3_SONNET_2024_02_29: LLMConfig(model="claude-3-sonnet-20240229", company='ANTHROPIC', input_cost=3, output_cost=15, client_kwargs={'max_completion_tokens': 4096}),
    LLMType.CLAUDE_3_HAIKU_2024_03_07: LLMConfig(model="claude-3-haiku-20240307", company='ANTHROPIC', input_cost=0.25, output_cost=1.25, client_kwargs={'max_completion_tokens': 4096}),

    # https://ai.google.dev/pricing#2_0flash
    LLMType.GEMINI_2_0_FLASH: LLMConfig(model="gemini-2.0-flash", company='GOOGLE', input_cost=0.1, output_cost=0.4),
    LLMType.GEMINI_2_0_FLASH_LITE_PREVIEW_02_05: LLMConfig(model="gemini-2.0-flash-lite-preview-02-05", company='GOOGLE', input_cost=0.075, output_cost=0.30),
    LLMType.GEMINI_1_5_FLASH: LLMConfig(model="gemini-1.5-flash", company='GOOGLE', input_cost=0.075, output_cost=0.3),
    LLMType.GEMINI_1_5_FLASH_8B: LLMConfig(model="gemini-1.5-flash-8b", company='GOOGLE', input_cost=0.0375, output_cost=0.15),
    LLMType.GEMINI_1_5_PRO: LLMConfig(model="gemini-1.5-pro", company='GOOGLE', input_cost=1.25, output_cost=5.00),
    # LLMType.GEMINI_1_0_PRO: LLMConfig(company='GOOGLE', input_cost=0.50, output_cost=1.50),

    LLMType.DEEPSEEK_R1: LLMConfig(model="deepseek-reasoner", company='DEEPSEEK', input_cost=0.14, output_cost=2.19, client_kwargs={'stream': False}),


    # https://x.ai/pricing
    LLMType.GROK_2_1212: LLMConfig(model="grok-2-1212", company='XAI', input_cost=2, output_cost=10, knowledge_cutoff_date="2024-07-17"),
}


if __name__ == "__main__":
    print(MODEL_CONFIGS[LLMType.GPT_4O])
    print(LLMType.GPT_4O)
    print(str(LLMType.GPT_4O))
    print(f"{LLMType.GPT_4O}")
    print(f"{LLMType.GPT_4O.value}")
    print(f"{LLMType.GPT_4O.name}")
 