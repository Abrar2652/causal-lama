from models.language_models import Llama2_7b, Llama3_8b, Llama2_13b, Qwen_7b, Qwen_14b, Qwen2_3b, Qwen2_7b, Mistral7B_RR, Zephyr_R2D2, Gemma2_9b
from models.generic_model import GenericHFModel


# Unique-family models loaded via the generic class (alias -> HF path).
# Used for cross-model RCS breadth (Table 1); diverse families, small/feasible.
GENERIC_MODELS = {
    'deepseek-r1-qwen-7b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    'gemma3-4b':           'google/gemma-3-4b-it',
    'phi35-mini':          'microsoft/Phi-3.5-mini-instruct',
    'ministral-8b':        'mistralai/Ministral-8B-Instruct-2410',
    'gpt-oss-20b':         'openai/gpt-oss-20b',
    'llama31-8b':          'meta-llama/Llama-3.1-8B-Instruct',
    'llama32-3b':          'meta-llama/Llama-3.2-3B-Instruct',
    'llama3-8b-gen':       'meta-llama/Meta-Llama-3-8B-Instruct',
}


def load_model(model_name, device, system_prompt='default'):

    models = {
        'llama2-7b': Llama2_7b,
        'llama3-8b': Llama3_8b,
        'llama2-13b': Llama2_13b,
        'qwen-7b': Qwen_7b,
        'qwen-14b': Qwen_14b,
        'qwen2-3b': Qwen2_3b,
        'qwen2-7b': Qwen2_7b,
        'mistral-7b-rr': Mistral7B_RR,
        'r2d2': Zephyr_R2D2,
        'gemma2-9b': Gemma2_9b
    }
    print(f"Initializing model: {model_name} on device: {device}")

    if model_name in GENERIC_MODELS:
        sp = None if system_prompt == 'default' else system_prompt
        return GenericHFModel(GENERIC_MODELS[model_name], device=device, system_prompt=sp)

    if model_name not in models:
        raise ValueError(f"Model {model_name} is not supported. Available: "
                         f"{list(models.keys()) + list(GENERIC_MODELS.keys())}")
    if system_prompt != 'default':
        return models[model_name](device=device, system_prompt=system_prompt)
    return models[model_name](device=device)
