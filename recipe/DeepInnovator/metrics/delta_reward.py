
import json
import re
import logging
import os
import asyncio
import random
import json_repair
from recipe.ResearchGAN.utils import call_agent  # Function to call agent
from recipe.ResearchGAN.utils import clean_idea  # Logger
# Configure logger
logger = logging.getLogger(__name__)
# Get log level from environment variable, default to WARN
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
IMPROVE_PROMPT = """
You will be given two research ideas proposed by two PhD students attempting to formulate a research hypothesis.
Additionally, you will be given a ground truth idea that has been peer-reviewed and validated.

## Task
Evaluate how close each of the two ideas is to the ground truth idea. Provide scores on a scale of 0-1, where higher scores indicate better alignment with the ground truth.
- IMPORTANT: Base judgment ONLY on content quality, NOT on format features (citations, DOI, formatting styles)
- **CRITICAL Red Flags for Fictional Ideas**:
* **Superficial Technical Depth**: Technical terms are mentioned but not meaningfully integrated; steps look detailed but lack executable specifics (e.g., missing computational complexity, data requirements, implementation challenges)
* **Missing Practical Challenges**: No discussion of computational complexity, data availability, measurement error, endogeneity, cross-country data harmonization, or other real-world implementation difficulties
* **Unrealistic Integration**: Multiple complex techniques are combined without explaining how they overcome fundamental incompatibilities or why the integration is theoretically sound
* **Abstract Limitations Only**: Limitations are described in abstract terms without engaging with specific empirical challenges or methodological tensions
* **Overly Ambitious Scope**: Simultaneously addresses too many disparate problems without acknowledging the complexity or potential conflicts

## Output Format
You must return a JSON object with the following structure:
{{
    "idea1_improve_score": <float between 0 and 1>,  // Higher score indicates better alignment with ground truth
    "idea2_improve_score": <float between 0 and 1>,  // Higher score indicates better alignment with ground truth
    "reason": "<explanation of your evaluation>"
}}

## Example
{{
    "idea1_improve_score": 0.8,
    "idea2_improve_score": 0.6,
    "reason": "Idea1 shows better alignment with the ground truth idea because it more closely matches the structure and content of the validated research approach."
}}

## Input
idea1: {idea1}
idea2: {idea2}
ground_truth: {ground_truth}

"""

def extract_all_ideas(conversation_history):
    """
    Extract all assistant's ideas from conversation history
    
    Args:
        conversation_history: List, each element is a dict containing "role" and "content" fields
        
    Returns:
        list: List containing all ideas, arranged chronologically (earliest first, latest last),
              each element is JSON-formatted idea content
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
                    # Try to extract improved_idea field, if not present use entire object
                    try:
                        idea_content = idea_json['improved_idea']
                    except (KeyError, TypeError):
                        idea_content = idea_json
                    ideas.append(idea_content)
                except json.JSONDecodeError:
                    # If standard parsing fails, try to repair with json_repair
                    try:
                        idea_json = json_repair.loads(idea_text)
                        try:
                            idea_content = idea_json['improved_idea']
                        except (KeyError, TypeError):
                            idea_content = idea_json
                        ideas.append(idea_content)
                    except Exception:
                        # If repair also fails, skip this idea
                        logger.debug(f"[delta_reward] Failed to parse idea JSON even with json_repair")
                        continue
    
    # Check if there are enough ideas
    if len(ideas) < 2:
        raise ValueError(f"Need at least 2 assistant ideas, but only found {len(ideas)}")
    
    return ideas


async def compute_score(data_source, messages, ground_truth, extra_info, **kwargs):
    # Convert messages to dict list format, refer to reward_function.py processing
    # Prefer delta_reward_kwargs, fall back to default_reward_kwargs if not available
    delta_reward_kwargs = kwargs.get("delta_reward_kwargs")
    if delta_reward_kwargs is None:
        # If no delta_reward_kwargs, use default_reward_kwargs
        default_reward_kwargs = kwargs.get("default_reward_kwargs", {})
        delta_reward_kwargs = default_reward_kwargs.copy()
    
    model = delta_reward_kwargs.pop("model", "qwen-plus-latest")

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
                logger.debug(f"[delta_reward] Failed to convert message item: {e}")
                continue
    
    try:
        all_ideas = extract_all_ideas(history)
    except Exception as e:
        logger.warning(f"[delta_reward] Failed to extract all ideas: {e}")
        return 0.0

    # If fewer than 2 ideas, cannot compare
    if len(all_ideas) < 2:
        logger.warning(f"[delta_reward] Need at least 2 ideas, but only found {len(all_ideas)}")
        return 0.0
    
    # Process ground_truth, if it's a string, try to parse
    ground_truth_parsed = ground_truth
    if isinstance(ground_truth, str):
        try:
            ground_truth_parsed = json.loads(ground_truth)
        except json.JSONDecodeError:
            # If parsing fails, keep as is (might be plain string)
            pass
    
    ground_truth_cleaned, _ = clean_idea(ground_truth_parsed)
    if isinstance(ground_truth_cleaned, dict):
        ground_truth_str = json.dumps(ground_truth_cleaned, ensure_ascii=False)
    else:
        ground_truth_str = str(ground_truth_cleaned)
    
    # Calculate sum of differences for all adjacent idea pairs
    total_score = 0.0
    
    # Iterate through all adjacent idea pairs (from old to new)
    for i in range(len(all_ideas) - 1):
        idea_new = all_ideas[i + 1]  # Newer idea
        idea_old = all_ideas[i]      # Older idea
        
        # If idea is string format, parse to dict first
        if isinstance(idea_new, str):
            try:
                idea_new = json.loads(idea_new)
            except json.JSONDecodeError:
                logger.warning(f"[delta_reward] Failed to parse idea_new[{i+1}] as JSON: {idea_new[:100]}")
                continue
        
        if isinstance(idea_old, str):
            try:
                idea_old = json.loads(idea_old)
            except json.JSONDecodeError:
                logger.warning(f"[delta_reward] Failed to parse idea_old[{i}] as JSON: {idea_old[:100]}")
                continue
        
        # Clean ideas
        new_idea_cleaned, _ = clean_idea(idea_new)
        old_idea_cleaned, _ = clean_idea(idea_old)
        
        # Ensure all ideas are JSON string format (for prompt)
        if isinstance(new_idea_cleaned, dict):
            new_idea_str = json.dumps(new_idea_cleaned, ensure_ascii=False)
        else:
            new_idea_str = str(new_idea_cleaned)
        
        if isinstance(old_idea_cleaned, dict):
            old_idea_str = json.dumps(old_idea_cleaned, ensure_ascii=False)
        else:
            old_idea_str = str(old_idea_cleaned)
        
        prompt = IMPROVE_PROMPT.format(idea1=new_idea_str, idea2=old_idea_str, ground_truth=ground_truth_str)
        
        try:
            # Call agent to generate response (async call)
            response = await call_agent(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **delta_reward_kwargs
            )
            
            # Try to extract JSON object from response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                # If response is API response object, extract content
                content = response.choices[0].message.content
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)
            
            # Try to parse JSON
            if isinstance(content, str):
                try:
                    response_dict = json_repair.loads(content)
                except Exception:
                    response_dict = json.loads(content)
            else:
                response_dict = content
            
            # Validate response format
            if not isinstance(response_dict, dict):
                logger.warning(f"[delta_reward] Response is not a dict for pair ({i}, {i+1}). Type: {type(response_dict)}. Response: {response_dict}")
                continue
            
            if "idea1_improve_score" not in response_dict or "idea2_improve_score" not in response_dict:
                logger.warning(
                    f"[delta_reward] Response missing required keys for pair ({i}, {i+1}). "
                    f"Keys: {response_dict.keys()}. Response: {response_dict}"
                )
                continue
            
            # Calculate difference score for this idea pair (new idea score - old idea score)
            pair_score = response_dict["idea1_improve_score"] - response_dict["idea2_improve_score"]
            
            # Validate score is numeric type
            if not isinstance(pair_score, (int, float)):
                logger.warning(f"[delta_reward] got invalid score type {type(pair_score)} for pair ({i}, {i+1}). Score: {pair_score}")
                continue
            
            # Accumulate to total score
            total_score += pair_score
            
        except Exception as e:
            # Catch exception when comparing single pair, log but continue processing other pairs
            logger.warning(f"[delta_reward] Error comparing idea pair ({i}, {i+1}): {e}")
            continue
    
    # Return sum of differences for all adjacent idea pairs
    return total_score