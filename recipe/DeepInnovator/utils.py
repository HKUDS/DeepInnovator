import re
from tokenize import Special
import json

from openai import OpenAI, AsyncOpenAI
from typing import Any, Union
import asyncio

async def call_agent(model: str, messages: list[dict[str, Any]], **kwargs):
    """
    Usage:
        response = await call_agent(
            model=self.user_model,
            messages=[{"role": "user", "content": prompt}],
            **self.user_model_kwargs, #api_key,api_base,...
        )
    """

    print(f"  🤖 Calling {model}...")

    # Prepare OpenAI client configuration
    # Convert api_base to base_url if present
    client_kwargs = {}
    if "api_base" in kwargs:
        client_kwargs["base_url"] = kwargs.pop("api_base")
    elif "base_url" in kwargs:
        client_kwargs["base_url"] = kwargs.pop("base_url")
    
    if "api_key" in kwargs:
        client_kwargs["api_key"] = kwargs.pop("api_key")

    if "num_retries" in kwargs:
        num_retries = kwargs.pop("num_retries")
    else:
        num_retries = 3
    # client = AsyncOpenAI(**client_kwargs)
    
    # for i in range(num_retries):
    #     try:
    #         response = await client.chat.completions.create(
    #             model=model,
    #             messages=messages,
    #             **kwargs
    #         )
    #         return response
    #     except Exception as e:
    #         if i == num_retries - 1:
    #             raise e
    #         else:
    #             await asyncio.sleep(2*(1+i))  # 等待2*i秒

    #             continue
    # Initialize OpenAI-compatible client
    # Use async with to ensure client is properly closed, avoiding resource leaks and connection errors
    async with AsyncOpenAI(**client_kwargs) as client:
        for i in range(num_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                return response
            except Exception as e:
                if i == num_retries - 1:
                    raise e
                else:
                    await asyncio.sleep(2*(1+i))  # Exponential backoff: wait 2*(i+1) seconds
                    continue
    
    # Should not reach here theoretically, but kept for type checking
    # If all retries fail, the raise e above will throw an exception
    raise RuntimeError("Unexpected: all retries failed but no exception was raised")

def clean_idea(idea):
    """Clean idea data, remove unwanted fields"""
    # If idea is not a dict type, return original value and empty dict
    need_keys = ['idea_summary', 'technical_approach']#, 'novelty_statement'

    if not isinstance(idea, dict):
        return idea, {}
        
    idea_pop = {}
    try:
        idea = idea.copy()
        
        try:
            idea = idea['improved_idea']
        except KeyError:
            pass
        all_keys = list(idea.keys())  # Convert to list to avoid modifying dict during iteration
        if 'current_limitations' in all_keys:
            all_keys.remove('current_limitations')
            try:
                original_value = idea['current_limitations']
                idea_pop['current_limitations'] = original_value
                # Clean bracket content
                idea['current_limitations'] = re.sub(r'\[[^\]]*\]', '', original_value)
            except Exception:
                pass  # Skip if processing fails

        # Iterate through all keys, remove fields not in need_keys

        for key in all_keys:
            if key not in need_keys:
                idea_pop[key] = idea.pop(key, None)
        # idea_pop['Cross-domain'] = idea.pop('Cross-domain', None)
        # idea_pop['source_paper_ids'] = idea.pop('source_paper_ids', None)
        # idea_pop['supporting_insights'] = idea.pop('supporting_insights', None)
        # idea_pop['confidence'] = idea.pop('confidence', None)

    except Exception:
        # If any exception occurs, return original value and empty dict
        return idea, {}
    
    return idea, idea_pop




def clean_json(data: Union[dict, list, Any], key_to_remove: str) -> Union[dict, list, Any]:
    """
    Recursively remove all matching keys from JSON data.
    
    Args:
        data: Can be dict, list, or other JSON-compatible types
        key_to_remove: Name of the key to remove
    
    Returns:
        Cleaned data (modified in place, but returns result for chaining)
    
    Example:
        >>> data = {"a": 1, "b": {"c": 2, "paper_path": "xxx"}, "paper_path": "yyy"}
        >>> clean_json(data, "paper_path")
        {'a': 1, 'b': {'c': 2}}
    """
    if isinstance(data, dict):
        # If dict, first recursively process all values
        for k, v in list(data.items()):
            # Recursively process nested values
            data[k] = clean_json(v, key_to_remove)
        
        # Then delete matching key
        if key_to_remove in data:
            del data[key_to_remove]
    
    elif isinstance(data, list):
        # If list, recursively process each element
        for i in range(len(data)):
            data[i] = clean_json(data[i], key_to_remove)
    
    # For other types (str, int, float, bool, None), return directly
    return data





# From CollabLLM: https://github.com/Wuyxin/collabllm
def is_valid_messages(msg: dict) -> bool:
    """
    check if is valid messages, including:
    1. <think> is paried with </think>
    2. is not empty inside and outside <think>
    3. is not nested, and at most one <think> block is allowed.
    4. can not be empty if remove ending "<|im_end|>"
    """
    content = msg.get("content")
    if not isinstance(content, str):
        return True

    # Base case: empty or whitespace-only content is invalid.
    if not content.strip():
        return False

    num_think_open = content.count("<think>")
    num_think_close = content.count("</think>")

    # Rule 1: Check for paired tags.
    if num_think_open != num_think_close:
        return False

    # Rule 3: Allow at most one think block.
    if num_think_open > 1:
        return False

    # Case 1: No <think> blocks.
    if num_think_open == 0:
        visible_content = content
    # Case 2: Exactly one <think> block.
    else:
        # Rule 2: Check for empty content inside the think block.
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if not match or not match.group(1).strip():
            return False

        # The "visible" content is what's outside the think block.
        visible_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    visible_content = visible_content.strip()

    # Rule 4 & 2 (outside): Check if visible content is empty after handling <|im_end|>.
    if visible_content.endswith("<|im_end|>"):
        visible_content = visible_content[: -len("<|im_end|>")]

    if not visible_content.strip():
        return False

    return True