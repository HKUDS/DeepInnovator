# Copyright 2026 DeepInnovator Ltd. and/or its affiliates
# Copyright 2024 CollabLLM Ltd. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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

import asyncio
import copy
import logging
import os
from typing import Any, Optional
from uuid import uuid4
import json
import json_repair

from recipe.DeepInnovator.utils import remove_think_block
from verl.interactions.base import BaseInteraction
from verl.utils.rollout_trace import rollout_trace_op
from recipe.DeepInnovator.utils import clean_idea

DISCRIMINATOR_PROMPT = """
Name: Idea Authenticity Checker
Description: Determine whether a given research idea comes from real, published research work or is a fictional/hypothetical research idea.

# Task Overview
1. Carefully analyze the provided research idea.
2. Make a judgment: 1 = real research work, 0 = fictional research idea.
4. Provide confidence level and detailed reasoning for the judgment.

# Important: Do NOT Rely on Format Features
- DO NOT use citation formats (e.g., arXiv citations, BibTeX format) to judge authenticity
- DO NOT use DOI numbers, paper IDs, or publication identifiers as indicators
- DO NOT rely on formatting styles (e.g., LaTeX formatting, citation styles) to make judgments
- Focus ONLY on the CONTENT QUALITY and SUBSTANCE of the research idea itself
- Format features can be easily fabricated and are not reliable indicators of authenticity
- Base your judgment solely on the technical depth, problem clarity, and limitations discussion in the content
- Be mindful to avoid excessive focus on details, as this is a research idea rather than a complete research proposal.

# Analysis Steps

## Step 1: Extract Key Components
- Identify the core problem being addressed
- Extract the main technical approach or methodology
- Note any specific methods, algorithms, or techniques mentioned
- Identify novelty claims and contributions
- Note any limitations or challenges discussed
- **CRITICAL**: Check the `technical_approach` field (if present) for:
* Whether steps are truly executable or just superficially detailed
* Presence of concrete implementation details (data sources, parameters, algorithms)
* Discussion of computational complexity or resource requirements
* Missing critical details that would be needed for actual implementation

## Step 2: Deep Technical Analysis
- **CRITICAL: Evaluate Technical Approach Executability**
* If `technical_approach` is present, check if steps are truly executable:
    - Do steps include concrete data sources, parameters, or algorithms?
    - Are computational complexity or resource requirements discussed?
    - Are there missing critical details needed for implementation?
    - Can a researcher actually follow these steps to reproduce results?
* **Red Flag**: Steps look detailed but lack executable specifics (e.g., "apply X method" without specifying how, when, or with what parameters)
- **CRITICAL: Check Technical Integration Depth**
* Verify if multiple techniques are meaningfully integrated or just superficially combined:
    - Are integration challenges acknowledged?
    - Is there explanation of how incompatible assumptions are resolved?
    - Are synergistic principles explained, or just stated?
* **Red Flag**: Multiple complex techniques combined without explaining fundamental incompatibilities or theoretical soundness

## Step 3: Make Judgment
- Consider all indicators together
- IMPORTANT: Base judgment ONLY on content quality, NOT on format features (citations, DOI, formatting styles)
- **CRITICAL Red Flags for Fictional Ideas**:
* **Superficial Technical Depth**: Technical terms are mentioned but not meaningfully integrated; steps look detailed but lack executable specifics (e.g., missing computational complexity, data requirements, implementation challenges)
* **Missing Practical Challenges**: No discussion of computational complexity, data availability, measurement error, endogeneity, cross-country data harmonization, or other real-world implementation difficulties
* **Unrealistic Integration**: Multiple complex techniques are combined without explaining how they overcome fundamental incompatibilities or why the integration is theoretically sound
* **Abstract Limitations Only**: Limitations are described in abstract terms without engaging with specific empirical challenges or methodological tensions
* **Overly Ambitious Scope**: Simultaneously addresses too many disparate problems without acknowledging the complexity or potential conflicts
- Real research work typically shows:
* High technical specificity with meaningful integration of concepts
* Well-defined problem context with clear boundaries
* Detailed limitations discussion including practical challenges (computational, data, methodological)
* Technical steps that are executable with concrete implementation details
* Acknowledgment of potential failures, edge cases, or methodological tensions
- Fictional ideas typically show:
* Low to medium technical specificity OR high specificity but superficial integration
* Vague problem context OR overly broad scope without clear boundaries
* Minimal limitations discussion OR abstract limitations without practical challenges
* Technical steps that look detailed but lack executable specifics
* Missing discussion of computational complexity, data requirements, or implementation challenges
- Make the authenticity judgment: 1 (real) or 0 (fictional)
- Your knowledge maybe outdated, so do not use your knowledge to judge if some methods are present or absent.

## Step 4: Assess Confidence
- High confidence (0.8-1.0): Clear indicators strongly support the judgment
- Medium confidence (0.5-0.8): Most indicators support the judgment, but some ambiguity exists
- Low confidence (0.0-0.5): Mixed indicators or insufficient information


## Step 5: Generate Reasoning
- List 3-5 key reasons supporting the judgment
- Reference specific aspects of the idea that led to the conclusion
- Be specific and concrete in the reasoning

# Output Requirements

## Authenticity Field
- Must be exactly 1 (real research work) or 0 (fictional research idea)
- Based on comprehensive evaluation of all indicators

## Confidence Field
- Number between 0.0 and 1.0
- Reflects how certain you are about the judgment
- Consider the strength and consistency of indicators

## Reasoning Field
- Array of 3-5 strings
- Each string should be a clear, specific reason
- Reference concrete aspects of the idea
- Explain why the judgment was made

## Here is a typical true idea:
{ground_truth}


# Input Data

## Research Idea
<idea>
{idea}
</idea>

Notes:
the result should be a valid JSON object, wrapped in <Judgment>...</Judgment>

Output Format:
<Judgment>
{{
  "authenticity": 0,
  "confidence": 0.95,
  "reasoning": ["The idea is not clearly from real research work", "The idea is not well-defined and has a clear technical approach", "The idea is fake and not executable with concrete implementation details"]
}}
</Judgment>
"""

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"

class DeepInnovatorInteraction(BaseInteraction):

    def __init__(self, config: dict):
        super().__init__(config)
        _config = copy.deepcopy(config)

        _config.pop("enable_log", None)

        self.name = _config.pop("name")

        self.termination_signal = _config.pop("termination_signal", TERMINATION_SIGNAL)
        self.num_retries = _config.pop("num_retries", 3)
        
        self.discriminator_model_kwargs = _config.pop("discriminator_kwargs", None)
        self.discriminator_model = self.discriminator_model_kwargs.pop("discriminator_model")

        self._instance_dict = {}

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        
        self.interaction_kwargs = kwargs
        
        return instance_id

    @rollout_trace_op
    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        assert messages[-1]["role"] in ["system", "assistant"], (
            "Last message input to the user model must be from system or assistant role"
        )

        from recipe.DeepInnovator.utils import call_agent
        
        try:
            idea = self._extract_last_idea(messages)
        except Exception as e:
            logger.exception(f"An unexpected error occurred in DeepInnovatorInteraction: {e}")
            return False, "You generated an invalid idea (Maybe a format error), please try again.", 0.0, {}
        
        ground_truth = self._instance_dict.get(instance_id, {}).get("ground_truth")
        
        prompt = DISCRIMINATOR_PROMPT.format(
            idea = idea,
            ground_truth = json.dumps(ground_truth)
        )
        
                
        response = ""
        full_response = None
        authenticity = None
        
        for i in range(self.num_retries):
            try:
                full_response = (
                    (
                        await call_agent(
                            model=self.discriminator_model,
                            messages=[{"role": "user", "content": prompt}],
                            **self.discriminator_model_kwargs,
                        )
                    )
                    .choices[0]
                    .message.content
                )
            except Exception as e:
                logger.exception(f"Retry {i} times, An unexpected error occurred in DeepInnovatorInteraction: {e}")
                continue

            try:
                if isinstance(full_response, str):
                    full_response = json_repair.loads(full_response)
            except Exception as e:
                logger.warning(f"Retry {i} times, [DeepInnovatorInteraction] Error extracting JSON: \n{e}\n. Retrying...")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {"authenticity", "confidence", "reasoning"}.issubset(keys):
                    authenticity = full_response.pop("authenticity")
                    authenticity_reasoning = full_response.pop("reasoning")
                    if isinstance(authenticity, int):
                        break
                    else:
                        logger.warning(
                            f"Retry {i} times, [DeepInnovatorInteraction] got an invaild full_response {full_response}. \
                                Retrying..."
                        )
                        continue
                else:
                    logger.warning(f"[DeepInnovatorInteraction] Keys {keys} do not match expected keys. Retrying...")
                    continue

        if full_response is None or authenticity is None:
            error_msg = f"Failed to get valid response from discriminator after {self.num_retries} retries. Please try again."
            logger.error(f"[DeepInnovatorInteraction] {error_msg}")
            return False, 'please try again', 0.0, {}
        if authenticity == 1:
            should_terminate_sequence = True
            if isinstance(authenticity_reasoning, (list, dict)):
                reason_str = json.dumps(authenticity_reasoning, ensure_ascii=False)
            else:
                reason_str = str(authenticity_reasoning)
            full_response['authenticity'] = "This is a real research idea. Reason: "+reason_str
        else:
            if isinstance(authenticity_reasoning, (list, dict)):
                reason_str = json.dumps(authenticity_reasoning, ensure_ascii=False)
            else:
                reason_str = str(authenticity_reasoning)
            full_response['authenticity'] = "This is a fictional research idea. Reason: "+reason_str

            should_terminate_sequence = False
        response = json.dumps(full_response, ensure_ascii=False)
        self._instance_dict[instance_id]["response"] = response
        
        if authenticity == 1:
            reward = 1.0
        else:
            reward = 0.0

        return should_terminate_sequence, response, reward, {}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


    def _extract_last_idea(self,messages):
        
        for message in reversed(messages):
            if message["role"] == "assistant":
                idea = json_repair.loads(message["content"])
                idea,_ = clean_idea(idea)
                
                return idea
            
        return None


    def _parse_messages(self, messages, strip_sys_prompt=True):
        if messages is None:
            return ""

        if strip_sys_prompt:
            messages = [msg for msg in messages if msg["role"] != "system"]

        messages = [remove_think_block(msg) for msg in messages]

        chat = "\n".join(f"**{m['role'].capitalize()}**: {m['content']}" for m in messages)

        return chat


def extract_json(s):
    
    def convert_value(value):
        true_values = {"true": True, "false": False, "null": None}
        value_lower = value.lower()
        
        if value_lower in true_values:
            return true_values[value_lower]
        
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value

    def parse_number(s, pos):
        start = pos
        while pos < len(s) and s[pos] in "-+0123456789.eE":
            pos += 1
        num_str = s[start:pos]
        try:
            if "." in num_str or "e" in num_str.lower():
                return float(num_str), pos
            else:
                return int(num_str), pos
        except ValueError:
            logger.error(f"Invalid number at position {start}: {num_str}")
            raise

    def skip_whitespace(s, pos):
        while pos < len(s) and s[pos] in " \t\n\r":
            pos += 1
        return pos

    def parse_string(s, pos):
        quote_char = s[pos]
        assert quote_char in ('"', "'")
        pos += 1
        result = ""
        
        while pos < len(s):
            c = s[pos]
            if c == "\\":
                pos += 1
                if pos >= len(s):
                    raise ValueError("Invalid escape sequence")
                c = s[pos]
                escape_sequences = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", quote_char: quote_char}
                result += escape_sequences.get(c, c)
            elif c == quote_char:
                pos += 1
                converted_value = convert_value(result)
                return converted_value, pos
            else:
                result += c
            pos += 1
        
        raise ValueError("Unterminated string")

    def parse_key(s, pos):
        pos = skip_whitespace(s, pos)
        if s[pos] in ('"', "'"):
            key, pos = parse_string(s, pos)
            return key, pos
        else:
            raise ValueError(f"Expected string for key at position {pos}. Input string: {s}")

    def parse_object(s, pos):
        obj = {}
        assert s[pos] == "{"
        pos += 1
        pos = skip_whitespace(s, pos)
        
        while pos < len(s) and s[pos] != "}":
            pos = skip_whitespace(s, pos)
            key, pos = parse_key(s, pos)
            pos = skip_whitespace(s, pos)
            
            if pos >= len(s) or s[pos] != ":":
                raise ValueError(f'Expected ":" at position {pos}. Input string: {s}')
            pos += 1
            pos = skip_whitespace(s, pos)
            
            value, pos = parse_value(s, pos)
            obj[key] = value
            pos = skip_whitespace(s, pos)
            
            if pos < len(s) and s[pos] == ",":
                pos += 1
                pos = skip_whitespace(s, pos)
            elif pos < len(s) and s[pos] == "}":
                break
            elif pos < len(s) and s[pos] != "}":
                raise ValueError(f'Expected "," or "}}" at position {pos}. Input string: {s}')
        
        if pos >= len(s) or s[pos] != "}":
            raise ValueError(f'Expected "}}" at position {pos}. Input string: {s}')
        pos += 1
        return obj, pos

    def parse_array(s, pos):
        lst = []
        assert s[pos] == "["
        pos += 1
        pos = skip_whitespace(s, pos)
        
        while pos < len(s) and s[pos] != "]":
            value, pos = parse_value(s, pos)
            lst.append(value)
            pos = skip_whitespace(s, pos)
            
            if pos < len(s) and s[pos] == ",":
                pos += 1
                pos = skip_whitespace(s, pos)
            elif pos < len(s) and s[pos] == "]":
                break
            elif pos < len(s) and s[pos] != "]":
                raise ValueError(f'Expected "," or "]" at position {pos}. Input string: {s}')
        
        if pos >= len(s) or s[pos] != "]":
            raise ValueError(f'Expected "]" at position {pos}. Input string: {s}')
        pos += 1
        return lst, pos

    def parse_triple_quoted_string(s, pos):
        if s[pos : pos + 3] == "'''":
            quote_str = "'''"
        elif s[pos : pos + 3] == '"""':
            quote_str = '"""'
        else:
            raise ValueError(f"Expected triple quotes at position {pos}. Input string: {s}")
        
        pos += 3
        result = ""
        
        while pos < len(s):
            if s[pos : pos + 3] == quote_str:
                pos += 3
                converted_value = convert_value(result)
                return converted_value, pos
            else:
                result += s[pos]
                pos += 1
        
        raise ValueError("Unterminated triple-quoted string")

    def parse_value(s, pos):
        pos = skip_whitespace(s, pos)
        
        if pos >= len(s):
            raise ValueError("Unexpected end of input")
        
        if s[pos] == "{":
            return parse_object(s, pos)
        elif s[pos] == "[":
            return parse_array(s, pos)
        elif s[pos : pos + 3] in ("'''", '"""'):
            return parse_triple_quoted_string(s, pos)
        elif s[pos] in ('"', "'"):
            return parse_string(s, pos)
        elif s[pos : pos + 4].lower() == "true":
            return True, pos + 4
        elif s[pos : pos + 5].lower() == "false":
            return False, pos + 5
        elif s[pos : pos + 4].lower() == "null":
            return None, pos + 4
        elif s[pos] in "-+0123456789.":
            return parse_number(s, pos)
        else:
            raise ValueError(f"Unexpected character at position {pos}: {s[pos]}. Input string: {s}")

    json_start = s.index("{")
    json_end = s.rfind("}")

    s = s[json_start : json_end + 1]

    s = s.strip()
    
    result, pos = parse_value(s, 0)
    
    pos = skip_whitespace(s, pos)
    
    if pos != len(s):
        logger.warning(
            f"JSON parsing incomplete at position {pos}/{len(s)}. "
            f"Remaining content: {s[pos:pos+100]}..."
        )
    
    return result
