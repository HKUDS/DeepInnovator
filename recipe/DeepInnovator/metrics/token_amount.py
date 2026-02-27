
# Copyright 2025 DeepInnovator team and/or its affiliates
# Copyright 2025 CollabLLM team and/or its affiliates
# Copyright 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import json_repair
import re
import os
import logging

# Configure logger
logger = logging.getLogger(__name__)
# Get log level from environment variable, default to WARN
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
def extract_latest_ideas(conversation_history):
    """
    Extract the latest two assistant's ideas from conversation history
    
    Args:
        conversation_history: List, each element is a dict containing "role" and "content" fields
        
    Returns:
        dict: Dictionary containing idea_1 (latest) and idea_2 (second-to-last), values are JSON-formatted idea content
    """
    # Store all found ideas
    ideas = []
    
    # Iterate through conversation history, find all assistant messages
    for item in conversation_history:
        if isinstance(item, dict) and item.get("role") == "assistant":
            content = item.get("content", "")
            
            # Use regex to extract content between <Idea>...</Idea>
            pattern = r'<Idea>(.*?)</Idea>'
            matches = re.findall(pattern, content, re.DOTALL)
            
            # Extract all matched ideas
            for match in matches:
                idea_text = match.strip()
                try:
                    # Try to parse JSON
                    idea_json = json.loads(idea_text)
                    ideas.append(idea_json)
                except json.JSONDecodeError:
                    # If standard parsing fails, try to repair with json_repair
                    try:
                        idea_json = json_repair.loads(idea_text)
                        ideas.append(idea_json)
                    except Exception:
                        # If repair also fails, skip this idea
                        logger.debug(f"[token_amount] Failed to parse idea JSON even with json_repair")
                        continue
    
    # Check if there are enough ideas
    if len(ideas) < 2:
        raise ValueError(f"Need at least 2 assistant ideas, but only found {len(ideas)}")
    try:
        idea_1 = ideas[-1]['improved_idea']
    except Exception as e:
        idea_1 = ideas[-1]
    try:
        idea_2 = ideas[-2]['improved_idea']
    except Exception as e:
        idea_2 = ideas[-2]


    # Return the latest two (idea_1 is latest, idea_2 is second-to-last)
    return {
        "idea_1": idea_1,  # Latest
        "idea_2": idea_2   # Second-to-last
    }


def compute_score(data_source, messages, ground_truth, extra_info, **kwargs):
    # prompt = extra_info["prompt"]

    # # Calculate the token penalty based on the length of the prompt
    # future_conv = messages[len(prompt) :]

    # # simple length estimation
    # total_tokens = sum(len(m.content.split()) for m in future_conv)
    history = []
    for item in messages:
        if isinstance(item, dict):
            history.append(item)
        else:
            # If object, try to access attributes and convert to dict
            try:
                history.append({
                    "role": getattr(item, "role", ""),
                    "content": getattr(item, "content", "")
                })
            except Exception as e:
                # If conversion fails, skip this item
                logger.debug(f"[token_amount] Failed to convert message item: {e}")
                continue

    latest_ideas = extract_latest_ideas(history)
    idea_1 = latest_ideas["idea_1"]
    idea_str = None
    # Convert the last idea to string uniformly before calculating length
    if isinstance(idea_1, str):
        idea_str = idea_1
    elif isinstance(idea_1, dict):
        # Dict serialized to string using JSON
        idea_str = json.dumps(idea_1, ensure_ascii=False)
    else:
        # Other types, fall back to plain string
        idea_str = str(idea_1)

    # Use character length as "length reward" metric, controlled within 3000-5000 range
    try:
        length = len(idea_str)
        if 3000 <= length <= 5000:
            # Within target range, give maximum reward
            total_tokens = 1.0
        elif length < 3000:
            # Less than 3000, give proportional reward
            total_tokens = length / 3000
        else:
            # Greater than 5000, penalize excess proportionally
            # Use linear decay: 1.0 at 5000, 0.0 at 10000
            total_tokens = max(0.0, 1.0 - (length - 5000) / 5000)
    except:
        total_tokens = 0.999

    return total_tokens
