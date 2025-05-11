from enum import Enum, auto
from typing import Optional
import os
from pydantic import BaseModel

class LLMType(str, Enum):
    
    # Open router models
    OPENROUTER_GPT_4O = auto()
    OPENROUTER_DEEPSEEK_CHAT_V3_0324 = auto()
    
    QWEN_2_5_CODER_32B_INSTRUCT = auto()
    QWEN_3_235B_A22B = auto()
    
    # Open source models
    QWEN_2_5_1_5B_INSTRUCT = auto()
    DEEPSEEK_R1_DISTILL_QWEN_14B = auto()
    QWEN_2_5_72B_INSTRUCT = auto()
    
    LLAMA_3_3_70B_INSTRUCT = auto()
    
    OPENROUTER_O4_MINI_HIGH = auto()
    OPENROUTER_CLAUDE_3_7_SONNET_THINKING = auto()
    OPENROUTER_GROK_3_MINI_BETA = auto()
    OPENROUTER_GROK_3_MINI_BETA_HIGH = auto()


    OPENROUTER_LLAMA_4_MAVERICK = auto()
    OPENROUTER_LLAMA_4_SCOUT = auto()

    OPENROUTER_INCEPTION_MERCURY_CODER_SMALL_BETA = auto()
    
    OPENROUTER_CLAUDE_3_5_HAIKU = auto()

    OPENROUTER_DEEPSEEK_CODER_V2 = auto()
    # qwen/qwq-32b
    OPENROUTER_QWQ_32B_HIGH = auto()
    OPENROUTER_NVIDIA_LLAMA_3_1_NEMOTRON_ULTRA_253B_V1_FREE = auto()
    OPENROUTER_NVIDIA_LLAMA_3_1_NEMOTRON_ULTRA_253B_V1_FREE_REASONING = auto()
    # OPENROUTER_CLAUDE_3_5_HAIKU = auto()
    OPENROUTER_GEMINI_2_5_FLASH_PREVIEW_THINKING = auto()
    OPENROUTER_QWEN_TURBO = auto()


    OPENROUTER_PHI_4 = auto()

    # deepseek/deepseek-prover-v2
    OPENROUTER_DEEPSEEK_PROVER_V2 = auto()

    # OpenAI models
    GPT_3_5_TURBO_0125 = auto()
    GPT_4_TURBO_2024_04_09 = auto()
    GPT_4O_MINI = auto()
    GPT_4O = auto()  #  = GPT_4O_2024_08_06
    GPT_4O_2024_11_20 = auto()
    GPT_4O_2024_08_06 = auto()
    GPT_4O_2024_05_13 = auto()
    GPT_4_5_PREVIEW = auto()
    GPT_4_1 = auto()
    GPT_4_1_MINI = auto()
    GPT_4_1_NANO = auto()
    
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
    
    O3 = auto()
    O3_HIGH = auto()
    # O1_MINI_HIGH = "o1-mini-2024-09-12"
    
    # Anthropic models https://docs.anthropic.com/en/docs/about-claude/models
    CLAUDE_3_7_SONNET_2025_02_19 = auto()
    CLAUDE_3_5_SONNET_2024_10_22 = auto()
    CLAUDE_3_5_HAIKU_2024_10_22 = auto()
    CLAUDE_3_OPUS_2024_02_29 = auto()
    CLAUDE_3_SONNET_2024_02_29 = auto()
    CLAUDE_3_HAIKU_2024_03_07 = auto()
    
    # Gemini models  https://ai.google.dev/gemini-api/docs/models/gemini
    # GEMINI_1_0_PRO = "gemini-1.0-pro"  # Deprecated on 2/15/2025
    GEMINI_1_5_FLASH = auto()
    GEMINI_1_5_FLASH_8B = auto()
    GEMINI_1_5_PRO = auto()
    GEMINI_2_0_FLASH = auto()
    GEMINI_2_0_FLASH_LITE_PREVIEW_02_05 = auto()
    GEMINI_2_5_FLASH_PREVIEW_04_17 = auto()
    GEMINI_2_5_PRO_PREVIEW_03_25 = auto()
    # XAI models
    GROK_2_1212 = auto()
    GROK_3_BETA = auto()
    GROK_3_MINI_BETA = auto()
    GROK_3_MINI_BETA_HIGH = auto()
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
    
    LLMType.OPENROUTER_GPT_4O: LLMConfig(model="openai/gpt-4o", company='OPENROUTER', input_cost=2.5, output_cost=10, knowledge_cutoff_date="2023-10", pretty_name="GPT-4o"),
    LLMType.OPENROUTER_DEEPSEEK_CHAT_V3_0324: LLMConfig(model="deepseek/deepseek-chat-v3-0324", company='OPENROUTER', input_cost=0.38, output_cost=0.89, knowledge_cutoff_date="2024-7", pretty_name="DeepSeek-Chat-V3"),
    LLMType.QWEN_2_5_CODER_32B_INSTRUCT: LLMConfig(model="qwen/qwen-2.5-coder-32b-instruct", company='OPENROUTER', input_cost=0.06, output_cost=0.15, knowledge_cutoff_date="2024-6", pretty_name="Qwen-2.5-Coder-32B"),
    LLMType.QWEN_3_235B_A22B: LLMConfig(model="qwen/qwen3-235b-a22b", company='OPENROUTER', input_cost=0.15, output_cost=0.6, knowledge_cutoff_date="2024-6", pretty_name="Qwen3-235B"),
    LLMType.LLAMA_3_3_70B_INSTRUCT: LLMConfig(model="meta-llama/llama-3.3-70b-instruct", company='OPENROUTER', input_cost=0.1, output_cost=0.25, knowledge_cutoff_date="2023-12", pretty_name="Llama-3.3-70B"),
    LLMType.OPENROUTER_O4_MINI_HIGH: LLMConfig(model="openai/o4-mini-high", company='OPENROUTER', input_cost=1.1, output_cost=4.4, knowledge_cutoff_date="2023-10", pretty_name="O4-Mini-High"),
    LLMType.OPENROUTER_CLAUDE_3_7_SONNET_THINKING: LLMConfig(model="anthropic/claude-3.7-sonnet:thinking", company='OPENROUTER', input_cost=3, output_cost=15, knowledge_cutoff_date="2024-10", pretty_name="Claude-3.7-Sonnet (Thinking)"),
    LLMType.OPENROUTER_GROK_3_MINI_BETA: LLMConfig(model="x-ai/grok-3-mini-beta", company='OPENROUTER', input_cost=0.3, output_cost=0.5, knowledge_cutoff_date="2024-11", pretty_name="Grok-3-Mini-Beta"),
    LLMType.OPENROUTER_GROK_3_MINI_BETA_HIGH: LLMConfig(model="x-ai/grok-3-mini-beta", company='OPENROUTER', input_cost=0.3, output_cost=0.5, knowledge_cutoff_date="2024-11", client_kwargs={'reasoning_effort': "high"}, pretty_name="Grok-3-Mini-Beta (High)"),
    LLMType.OPENROUTER_LLAMA_4_MAVERICK: LLMConfig(model="meta-llama/llama-4-maverick", company='OPENROUTER', input_cost=0.17, output_cost=0.6, knowledge_cutoff_date="2024-8", pretty_name="Llama-4-Maverick"),
    LLMType.OPENROUTER_LLAMA_4_SCOUT: LLMConfig(model="meta-llama/llama-4-scout", company='OPENROUTER', input_cost=0.08, output_cost=0.3, knowledge_cutoff_date="2024-8", pretty_name="Llama-4-Scout"),
    LLMType.OPENROUTER_INCEPTION_MERCURY_CODER_SMALL_BETA: LLMConfig(model="inception/mercury-coder-small-beta", company='OPENROUTER', input_cost=0.25, output_cost=1, knowledge_cutoff_date="2024-11", pretty_name="Mercury-Coder-Small-Beta"),
    LLMType.OPENROUTER_CLAUDE_3_5_HAIKU: LLMConfig(model="anthropic/claude-3.5-haiku", company='OPENROUTER', input_cost=0.8, output_cost=4, knowledge_cutoff_date="2024-7", pretty_name="Claude-3.5-Haiku"),
    LLMType.OPENROUTER_DEEPSEEK_CODER_V2: LLMConfig(model="deepseek/deepseek-coder", company='OPENROUTER', input_cost=0.04, output_cost=0.12, knowledge_cutoff_date="2023-11", pretty_name="DeepSeek-Coder"),
    LLMType.OPENROUTER_QWQ_32B_HIGH: LLMConfig(model="qwen/qwq-32b", company='OPENROUTER', input_cost=0.15, output_cost=0.2, knowledge_cutoff_date="2024-7", client_kwargs={'reasoning_effort': "high"}, pretty_name="QWQ-32B (High)"),
    LLMType.OPENROUTER_NVIDIA_LLAMA_3_1_NEMOTRON_ULTRA_253B_V1_FREE: LLMConfig(model="nvidia/llama-3.1-nemotron-ultra-253b-v1:free", company='OPENROUTER', input_cost=0, output_cost=0, knowledge_cutoff_date="2023-12", client_kwargs={'reasoning_effort': "high"}, pretty_name="Llama-3.1-Nemotron-Ultra-253B-V1"),
    LLMType.OPENROUTER_NVIDIA_LLAMA_3_1_NEMOTRON_ULTRA_253B_V1_FREE_REASONING: LLMConfig(model="nvidia/llama-3.1-nemotron-ultra-253b-v1:free", company='OPENROUTER', input_cost=0, output_cost=0, knowledge_cutoff_date="2023-12", client_kwargs={'reasoning_effort': "high"}, pretty_name="Llama-3.1-Nemotron-Ultra-253B-V1-Reasoning"),
    LLMType.OPENROUTER_GEMINI_2_5_FLASH_PREVIEW_THINKING: LLMConfig(model="google/gemini-2.5-flash-preview:thinking", company='OPENROUTER', input_cost=0.15, output_cost=3.5, knowledge_cutoff_date="2025-01", client_kwargs={'reasoning_effort': "high"}, pretty_name="Gemini-2.5-Flash-Preview (Thinking)"),
    LLMType.OPENROUTER_PHI_4: LLMConfig(model="microsoft/phi-4", company='OPENROUTER', input_cost=0.07, output_cost=0.14, knowledge_cutoff_date="2023-10", pretty_name="Phi-4"),
    LLMType.OPENROUTER_DEEPSEEK_PROVER_V2: LLMConfig(model="deepseek/deepseek-prover-v2", company='OPENROUTER', input_cost=0.7, output_cost=2.18, knowledge_cutoff_date="2024-7", client_kwargs={'reasoning_effort': "high"}, pretty_name="DeepSeek-Prover-V2"),
    LLMType.QWEN_2_5_1_5B_INSTRUCT: LLMConfig(model="Qwen/Qwen2.5-1.5B-Instruct", company='VLLM', input_cost=0, output_cost=0, knowledge_cutoff_date="2023-12", pretty_name="Qwen-2.5-1.5B-Instruct"),
    LLMType.DEEPSEEK_R1_DISTILL_QWEN_14B: LLMConfig(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", company='VLLM', input_cost=0, output_cost=0, knowledge_cutoff_date="2023-12", pretty_name="DeepSeek-R1-Distill-Qwen-14B"),
    LLMType.QWEN_2_5_72B_INSTRUCT: LLMConfig(model="Qwen/Qwen2.5-72B-Instruct", company='VLLM', input_cost=0, output_cost=0, knowledge_cutoff_date="2023-12", pretty_name="Qwen-2.5-72B-Instruct"),
    LLMType.OPENROUTER_QWEN_TURBO: LLMConfig(model="qwen/qwen-turbo", company='OPENROUTER', input_cost=0.05, output_cost=0.2, knowledge_cutoff_date="2023-9", pretty_name="Qwen-2.5-Turbo"),
    
    # https://platform.openai.com/docs/pricing
    LLMType.GPT_3_5_TURBO_0125: LLMConfig(model="gpt-3.5-turbo-0125", company='OPENAI', input_cost=0.5, output_cost=1.5, knowledge_cutoff_date="2022-01", pretty_name="GPT-3.5-Turbo-0125"),
    LLMType.GPT_4_TURBO_2024_04_09: LLMConfig(model="gpt-4-turbo-2024-04-09", company='OPENAI', input_cost=10, output_cost=30, knowledge_cutoff_date="2023-12", pretty_name="GPT-4-Turbo-2024-04-09"),
    LLMType.GPT_4O_MINI: LLMConfig(model="gpt-4o-mini", company='OPENAI', input_cost=0.15, output_cost=0.6, knowledge_cutoff_date="2023-10", pretty_name="GPT-4o-Mini"),
    LLMType.GPT_4O: LLMConfig(model="gpt-4o", company='OPENAI', input_cost=2.5, output_cost=10, knowledge_cutoff_date="2023-10", pretty_name="GPT-4o"),
    LLMType.GPT_4O_2024_11_20: LLMConfig(model="gpt-4o-2024-11-20", company='OPENAI', input_cost=2.5, output_cost=10, knowledge_cutoff_date="2023-10", pretty_name="GPT-4o"),
    LLMType.GPT_4O_2024_08_06: LLMConfig(model="gpt-4o-2024-08-06", company='OPENAI', input_cost=2.5, output_cost=10, knowledge_cutoff_date="2023-10", pretty_name="GPT-4o"),
    LLMType.GPT_4O_2024_05_13: LLMConfig(model="gpt-4o-2024-05-13", company='OPENAI', input_cost=5, output_cost=15, knowledge_cutoff_date="2023-10", pretty_name="GPT-4o"),
    LLMType.GPT_4_1: LLMConfig(model="gpt-4.1-2025-04-14", company='OPENAI', input_cost=2, output_cost=8, knowledge_cutoff_date="2023-12", pretty_name="GPT-4.1"),
    LLMType.GPT_4_1_MINI: LLMConfig(model="gpt-4.1-mini-2025-04-14", company='OPENAI', input_cost=0.4, output_cost=1.6, knowledge_cutoff_date="2023-12", pretty_name="GPT-4.1-Mini"),
    LLMType.GPT_4_1_NANO: LLMConfig(model="gpt-4.1-nano-2025-04-14", company='OPENAI', input_cost=0.1, output_cost=0.4, knowledge_cutoff_date="2023-12", pretty_name="GPT-4.1-Nano"),
    LLMType.O1_MINI: LLMConfig(model="o1-mini-2024-09-12", company='OPENAI', input_cost=1.1, output_cost=4.4, knowledge_cutoff_date="2023-10", client_kwargs={'temperature': 1}, pretty_name="O1-Mini"),
    LLMType.O3: LLMConfig(model="o3-2025-04-16", company='OPENAI', input_cost=10, output_cost=40, knowledge_cutoff_date="2023-10", client_kwargs={'temperature': 1}, pretty_name="O3"),
    LLMType.O3_HIGH: LLMConfig(model="o3-2025-04-16", company='OPENAI', input_cost=10, output_cost=40, knowledge_cutoff_date="2023-10", client_kwargs={'temperature': 1, 'reasoning': {"effort": "high"}}, pretty_name="O3 (High)"),
    # LLMType.O3_MINI_2025_01_31_HIGH: LLMConfig(company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1, 'reasoning': {"effort": "high"}}),
    LLMType.GPT_4_5_PREVIEW: LLMConfig(model="gpt-4.5-preview", company='OPENAI', input_cost=75, output_cost=150, knowledge_cutoff_date="2023-10", pretty_name="GPT-4.5-Preview"),

    LLMType.O3_MINI_HIGH: LLMConfig(model="o3-mini-2025-01-31", company='OPENAI', input_cost=1.1, output_cost=4.4, knowledge_cutoff_date="2023-10", client_kwargs={'temperature': 1, 'reasoning': {"effort": "high"}}, pretty_name="O3-Mini (High)"),
    LLMType.O3_MINI_2025_01_31: LLMConfig(model="o3-mini-2025-01-31", company='OPENAI', input_cost=1.1, output_cost=4.4, knowledge_cutoff_date="2023-10", client_kwargs={'temperature': 1}, pretty_name="O3-Mini"),
    LLMType.O3_MINI: LLMConfig(model="o3-mini-2025-01-31", company='OPENAI', input_cost=1.1, output_cost=4.4, knowledge_cutoff_date="2023-10", client_kwargs={'temperature': 1}, pretty_name="O3-Mini"),
    LLMType.O1_PREVIEW_2024_09_12: LLMConfig(model="o1-preview-2024-09-12", company='OPENAI', input_cost=15, output_cost=60, knowledge_cutoff_date="2023-10", client_kwargs={'temperature': 1}, pretty_name="O1-Preview"),
    LLMType.O1: LLMConfig(model="o1-2024-12-17", company='OPENAI', input_cost=15, output_cost=60, knowledge_cutoff_date="2023-10", client_kwargs={'temperature': 1}, pretty_name="O1"),
    LLMType.O1_HIGH: LLMConfig(model="o1-2024-12-17", company='OPENAI', input_cost=15, output_cost=60, knowledge_cutoff_date="2023-10", client_kwargs={'temperature': 1, 'reasoning': {"effort": "high"}}, pretty_name="O1 (High)"),
    LLMType.O1_MINI: LLMConfig(model="o1-mini-2024-09-12", company='OPENAI', input_cost=1.1, output_cost=4.4, knowledge_cutoff_date="2023-10", client_kwargs={'temperature': 1}, pretty_name="O1-Mini"),
    # LLMType.O1_MINI_HIGH: LLMConfig(company='OPENAI', input_cost=1.1, output_cost=4.4, client_kwargs={'temperature': 1, 'reasoning': {"effort": "high"}}),
    
    # Anthropic models https://www.anthropic.com/pricing#anthropic-api
    # https://docs.anthropic.com/en/docs/about-claude/models
    LLMType.CLAUDE_3_7_SONNET_2025_02_19: LLMConfig(model="claude-3-7-sonnet-20250219", company='ANTHROPIC', input_cost=3, output_cost=15, knowledge_cutoff_date="2024-11", client_kwargs={'max_completion_tokens': 8192}, pretty_name="Claude-3.7-Sonnet"),
    LLMType.CLAUDE_3_5_SONNET_2024_10_22: LLMConfig(model="claude-3-5-sonnet-20241022", company='ANTHROPIC', input_cost=3, output_cost=15, knowledge_cutoff_date="2024-04", client_kwargs={'max_completion_tokens': 8192}, pretty_name="Claude-3.5-Sonnet"),
    LLMType.CLAUDE_3_5_HAIKU_2024_10_22: LLMConfig(model="claude-3-5-haiku-20241022", company='ANTHROPIC', input_cost=0.8, output_cost=4, knowledge_cutoff_date="2024-07", client_kwargs={'max_completion_tokens': 8192}, pretty_name="Claude-3.5-Haiku"),
    LLMType.CLAUDE_3_OPUS_2024_02_29: LLMConfig(model="claude-3-opus-20240229", company='ANTHROPIC', input_cost=15, output_cost=75, knowledge_cutoff_date="2023-08", client_kwargs={'max_completion_tokens': 4096}, pretty_name="Claude-3-Opus"),
    LLMType.CLAUDE_3_SONNET_2024_02_29: LLMConfig(model="claude-3-sonnet-20240229", company='ANTHROPIC', input_cost=3, output_cost=15, knowledge_cutoff_date="2023-08", client_kwargs={'max_completion_tokens': 4096}, pretty_name="Claude-3-Sonnet"),
    LLMType.CLAUDE_3_HAIKU_2024_03_07: LLMConfig(model="claude-3-haiku-20240307", company='ANTHROPIC', input_cost=0.25, output_cost=1.25, knowledge_cutoff_date="2023-08", client_kwargs={'max_completion_tokens': 4096}, pretty_name="Claude-3-Haiku"),

    # https://ai.google.dev/pricing#2_0flash
    LLMType.GEMINI_2_0_FLASH: LLMConfig(model="gemini-2.0-flash", company='GOOGLE', input_cost=0.1, output_cost=0.4, knowledge_cutoff_date="2024-06", pretty_name="Gemini-2.0-Flash"),
    LLMType.GEMINI_2_0_FLASH_LITE_PREVIEW_02_05: LLMConfig(model="gemini-2.0-flash-lite-preview-02-05", company='GOOGLE', input_cost=0.075, output_cost=0.30, knowledge_cutoff_date="2025-01", pretty_name="Gemini-2.0-Flash-Lite"),
    LLMType.GEMINI_1_5_FLASH: LLMConfig(model="gemini-1.5-flash", company='GOOGLE', input_cost=0.075, output_cost=0.3, knowledge_cutoff_date="2024-05", pretty_name="Gemini-1.5-Flash"),
    LLMType.GEMINI_1_5_FLASH_8B: LLMConfig(model="gemini-1.5-flash-8b", company='GOOGLE', input_cost=0.0375, output_cost=0.15, knowledge_cutoff_date="2024-05", pretty_name="Gemini-1.5-Flash-8B"),
    LLMType.GEMINI_1_5_PRO: LLMConfig(model="gemini-1.5-pro", company='GOOGLE', input_cost=1.25, output_cost=5.00, knowledge_cutoff_date="2024-05", pretty_name="Gemini-1.5-Pro"),
    # LLMType.GEMINI_1_0_PRO: LLMConfig(company='GOOGLE', input_cost=0.50, output_cost=1.50, knowledge_cutoff_date="2023-02"),
    LLMType.GEMINI_2_5_FLASH_PREVIEW_04_17: LLMConfig(model="gemini-2.5-flash-preview-04-17", company='GOOGLE', input_cost=0.15, output_cost=0.6, client_kwargs={'reasoning_effort': "high"}, knowledge_cutoff_date="2025-01", pretty_name="Gemini-2.5-Flash"),
    LLMType.GEMINI_2_5_PRO_PREVIEW_03_25: LLMConfig(model="gemini-2.5-pro-preview-03-25", company='GOOGLE', input_cost=1.25, output_cost=10, knowledge_cutoff_date="2025-01", pretty_name="Gemini-2.5-Pro"),
    
    LLMType.DEEPSEEK_R1: LLMConfig(model="deepseek-reasoner", company='DEEPSEEK', input_cost=0.14, output_cost=2.19, knowledge_cutoff_date="2024-10", client_kwargs={'stream': False}, pretty_name="DeepSeek-R1"),


    # https://x.ai/pricing
    LLMType.GROK_2_1212: LLMConfig(model="grok-2-1212", company='XAI', input_cost=2, output_cost=10, knowledge_cutoff_date="2024-07-17", pretty_name="Grok-2"),
    LLMType.GROK_3_BETA: LLMConfig(model="grok-3-beta", company='XAI', input_cost=3, output_cost=15, knowledge_cutoff_date="2024-11-17", pretty_name="Grok-3-Beta"),
    LLMType.GROK_3_MINI_BETA: LLMConfig(model="grok-3-mini-beta", company='XAI', input_cost=0.3, output_cost=0.5, knowledge_cutoff_date="2024-11", pretty_name="Grok-3-Mini-Beta"),
    LLMType.GROK_3_MINI_BETA_HIGH: LLMConfig(model="grok-3-mini-beta", company='XAI', input_cost=0.3, output_cost=0.5, knowledge_cutoff_date="2024-11", client_kwargs={'reasoning_effort': "high"}, pretty_name="Grok-3-Mini-Beta (High)"),
}


if __name__ == "__main__":
    print(MODEL_CONFIGS[LLMType.GPT_4O])
    print(LLMType.GPT_4O)
    print(str(LLMType.GPT_4O))
    print(f"{LLMType.GPT_4O}")
    print(f"{LLMType.GPT_4O.value}")
    print(f"{LLMType.GPT_4O.name}")
 