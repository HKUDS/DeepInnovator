"""Utility functions for data preparation pipeline."""
import os
import json
import re
from typing import Any, Optional, Sequence, List
import logging
import sys
import time
import base64
from datetime import datetime, date               # For timestamp management and key dates
from itertools import count as count_iterator  # For generating unique IDs
from pathlib import Path                    # For cross-platform file path handling
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party library imports for configuration management and API communication
from dotenv import load_dotenv              # For loading environment variables from .env files
from omegaconf import OmegaConf             # For hierarchical configuration management
from openai import OpenAI                   # For OpenAI-compatible API communication
import feedparser
import requests
from dateutil import parser as date_parser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PyPDF2 import PdfReader

load_dotenv()

logger = logging.getLogger(__name__)


def clean_idea(idea):
    idea_pop = {}
    idea = idea.copy()
    idea_pop['Cross-domain'] = idea.pop('Cross-domain', None)
    idea_pop['source_paper_ids'] = idea.pop('source_paper_ids', None)
    idea_pop['supporting_insights'] = idea.pop('supporting_insights', None)
    idea_pop['confidence'] = idea.pop('confidence', None)
    try:
        idea_pop['current_limitations'] = idea['current_limitations']
        idea['current_limitations'] = re.sub(r'\[[^\]]*\]', '', idea['current_limitations'])
    except:
        return idea
    return idea, idea_pop


class Paper_Profile():
    def __init__(self, 
    memory_path: str = 'data/paper_memory.json', 
    group_path: str = 'data/paper_group.json',
    create_empty: bool = False):
        self.memory_path = memory_path
        self.group_path = group_path
        self.paper_memory = self.create_empty_paper_memory()
        self.paper_group = []
        if create_empty:       
            self.save_paper_memory()
            self.save_paper_group()
        else:
            self.read_paper_memory()
            self.read_paper_group()
    def read_paper_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                paper_memory = json.load(f)
            if 'papers' not in paper_memory:
                print(f"Paper memory format is incorrect, {self.memory_path}")
                print("use create_empty_paper_memory() to create a new paper memory")
                self.save_paper_memory()
                return None
            self.paper_memory = paper_memory
            return paper_memory
        else:
            print(f"Paper profile not found, {self.memory_path}")
            return None
    def save_paper_memory(self):
        dir_name = os.path.dirname(self.memory_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.paper_memory, f, ensure_ascii=False, indent=2)
        print(f"Paper profile saved to {self.memory_path}")
    def create_empty_paper_memory(self):
        self.paper_memory = {
            'type': 'paper_memory',        # Profile type identifier
            'papers': [],                   # List to accumulate processed paper memories
            'memories': [],                 # List to accumulate memories
            'idea_sparks': [],              # List to accumulate generated insights
            'stats': {                      # Processing statistics
                'total_papers': 0,          # Total papers processed
                'total_connections': 0,     # Total paper connections generated
                'total_serendipity': 0      # Total serendipity insights generated
            }}
        
        return self.paper_memory
    def read_paper_group(self):
        if os.path.exists(self.group_path):
            with open(self.group_path, 'r', encoding='utf-8') as f:
                paper_group = json.load(f)
            self.paper_group = paper_group
            return paper_group
        else:
            print(f"Paper group not found, {self.group_path}")
            return None
    def save_paper_group(self):
        dir_name = os.path.dirname(self.group_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(self.group_path, 'w', encoding='utf-8') as f:
            json.dump(self.paper_group, f, ensure_ascii=False, indent=2)
        print(f"Paper group saved to {self.group_path}")
    def create_empty_paper_group(self):
        self.paper_group = []
        return self.paper_group
    def add_paper_memory(self, this_paper_memory: dict):
        this_paper_id = this_paper_memory["paper_id"]
        if this_paper_id not in self.paper_memory["papers"]:
            self.paper_memory["papers"].append(this_paper_id)
            self.paper_memory["memories"].append(this_paper_memory)
            self.save_paper_memory()
            return True
        elif this_paper_id in self.paper_memory["papers"] and this_paper_id not in self.all_paper_id_in_group():
            return True

        else:
            print("already saved in paper profile!")
            return False
    def add_paper_group(self, this_paper_group: dict):
        self.paper_group.append(this_paper_group)
        self.save_paper_group()
    def update_paper_group(self, this_paper_group: dict):
        for group in self.paper_group:
            if group['group_id'] == this_paper_group['group_id']:
                group['group_content'].append(this_paper_group['group_content'])
                group['group_description'] = this_paper_group['group_description']
                self.save_paper_group()
                return True
        return False
    def all_paper_id_in_group(self):
        all_paper_id = []
        for group in self.paper_group:
            for paper in group['group_content']:
                # try:
                all_paper_id.append(paper['paper_id'])
                # except:
                #     print(f"paper_id not found in group_content: {paper}")
                #     exit()
        return all_paper_id
    def info(self):
        print(f"Paper Memory: {self.paper_memory.keys()}")
        print(f"Paper group: {self.paper_group}")
        print(f"Total papers: {len(self.paper_memory['papers'])}")
        print(f"Total memories: {len(self.paper_memory['memories'])}")
        print(f"Total idea sparks: {len(self.paper_memory['idea_sparks'])}")
        print(f"Total groups: {len(self.paper_group)}")



class Paper():
    def __init__(self, paper_id: str,paper_md_path,paper_memory_path):
        self.paper_id = paper_id
        self.paper_md_path = paper_md_path
        self.paper_memory_path = paper_memory_path
        self.paper_memory = {}
        self.paper_content = ''
        self._init_paper_memory()
        self._init_paper_content()
    def _init_paper_memory(self):
        with open(os.path.join(self.paper_memory_path, f"{self.paper_id}.json"), 'r', encoding='utf-8') as f:
            self.paper_memory = json.load(f)
    def _init_paper_content(self):
        with open(os.path.join(self.paper_md_path, f"{self.paper_id}.md"), 'r', encoding='utf-8') as f:
            self.paper_content = f.read()
    def get_paper_content(self):
        return self.paper_content
    def get_paper_memory(self):
        return self.paper_memory
    def get_paper_id(self):
        return self.paper_id


def call_agent(agent_name: str, prompt: str, config):
    """
    Generic function to call any configured agent (regular or insight) with a given prompt.

    INPUT:
        agent_name: str - Name of the agent to call (e.g., "paper_router", "paper_connections")
        prompt: str - Formatted prompt string to send to the agent
        config: Dict - Loaded configuration containing agent and model definitions

    OUTPUT:
        Dict or {} - Parsed JSON response from the agent, or empty dict on failure

    GOAL:
        Provide a unified interface for calling different types of agents (regular agents
        and insight agents) with proper error handling, model selection, and response parsing.
    """
    try:
        # Determine agent type and configuration
        if agent_name in config['agents']:
            # Regular agent (paper_router, memory_merger, etc.)
            agent_config = config['agents'][agent_name]
            model_set = agent_config['model']  # Get model set from agent config
        elif 'insights' in config['agents'] and agent_name in config['agents']['insights']:
            # Insight agent (paper_connections, serendipity_engine, etc.)
            agent_config = config['agents']['insights'][agent_name]
            # Insight agents use the main/big model set by default
            model_set = 'model_set'
        else:
            # Agent not found in configuration
            print(f"  ❌ Agent {agent_name} not found in configuration")
            return {}

        # Resolve model set to actual model list
        model_list = config['models']['model_sets'][model_set]
        if isinstance(model_list, dict) and 'models' in model_list:
            model_list = model_list['models']
        # Use first model in the set (failover would be handled by SafeAgent in production)
        model_name = model_list[0]
        model_config = config['_processed_models'][model_name]

        print(f"  🤖 Calling {agent_name} with model {model_config['model_name']}...")

        # Prepare OpenAI client configuration
        client_kwargs = {}
        if model_config.get("base_url"):
            client_kwargs["base_url"] = model_config["base_url"]
        if model_config.get("api_key"):
            client_kwargs["api_key"] = model_config["api_key"]

        # Initialize OpenAI-compatible client
        client = OpenAI(**client_kwargs)

        # Call agent with streaming output
        response = client.chat.completions.create(
            model=model_config['model_name'],
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        # Process streaming response and extract JSON
        result_text = process_stream_output(response)["output"]
        result_json = extract_json_from_markdown(result_text)

        # Handle JSON parsing failures
        if result_json is None:
            print(f"  ⚠️  Failed to extract JSON from response")
            print(f"  Full response:\n{result_text}")
            return {}

        return result_json
    except Exception as e:
        # Comprehensive error handling with traceback
        print(f"  ❌ Error calling {agent_name}: {e}")
        import traceback
        traceback.print_exc()
        return {}


def load_config():
    """
    Load configuration from YAML files.
    
    Returns:
        Dictionary containing agents, models, and processed model configurations
    """
    config_dir = Path(__file__).parent.parent / "config"
    
    # Load agents
    agents = {}
    agents_dir = config_dir / "agents"
    if agents_dir.exists():
        for yaml_file in agents_dir.glob("*.yaml"):
            agent_name = yaml_file.stem
            agents[agent_name] = OmegaConf.load(yaml_file)
    
    # Load models
    models = {}
    models_dir = config_dir / "models"
    if models_dir.exists():
        for yaml_file in models_dir.glob("*.yaml"):
            config_name = yaml_file.stem
            config_data = OmegaConf.load(yaml_file)
            # Convert to dict for easier access
            models[config_name] = OmegaConf.to_container(config_data, resolve=True)
    
    # Process models: merge provider info into model configs
    _processed_models = {}
    providers = models.get("providers", {})
    model_defs = models.get("models", {})
    
    for model_name, model_config in model_defs.items():
        processed = dict(model_config)  # Copy dict
        provider_name = processed.get("provider")
        if provider_name and provider_name in providers:
            provider_config = providers[provider_name]
            # Merge provider config into model config
            processed.update({
                "base_url": provider_config.get("base_url"),
                "api_key": provider_config.get("api_key"),
            })
        _processed_models[model_name] = processed
    
    return {
        "agents": agents,
        "models": models,
        "_processed_models": _processed_models,
    }


def setup_logger(log_file: str = None,logger_name: str = 'default_logger', level: int = logging.INFO):
    """
    Setup logger with both console and file handlers.
    
    INPUT:
        log_file: str - Optional path to log file. If None, only console logging.
        logger_name: str - Name of the logger (default: 'default_logger')
        level: int - Logging level (default: logging.INFO)
    
    OUTPUT:
        logging.Logger - Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def extract_json_from_markdown(markdown_text: str) -> dict[str, Any] | None:
    """Extract JSON from markdown code blocks or raw text.

    Args:
        markdown_text: Text that may contain JSON (in code blocks or raw)

    Returns:
        Parsed JSON dict, or None if parsing fails
    """
    # Try patterns in order of specificity
    patterns = [
        # Standard markdown code block with json label
        r"```(?:json\n|json\s*)?\n(.*?)\n```",
        # Code block without language specification
        r"```\n(.*?)\n```",
        # JSON wrapped in single backticks
        r"`({.*?})`",
        # Bare JSON object (fallback)
        r'({[\s\S]*"[^"]*"\s*:[\s\S]*})',
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, markdown_text, re.DOTALL)

        for match in matches:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                # This match didn't work, continue to next match
                continue

    # Try parsing entire text as JSON
    try:
        return json.loads(markdown_text)
    except (json.JSONDecodeError, ValueError):
        return None


def process_stream_output(
    output_generator,
    print_process: bool = False,
    incremental: bool = True,
) -> dict[str, str]:
    """Process streaming output from OpenAI API.

    Args:
        output_generator: Generator from OpenAI API
        print_process: Whether to print output as it streams
        incremental: Whether output is incremental

    Returns:
        Dict with 'output' key containing accumulated text
    """
    res = ""

    for chunk in output_generator:
        try:
            # OpenAI streaming response structure
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
            else:
                continue

            if content == "":
                continue

            if incremental:
                if print_process:
                    print(content, end="", flush=True)
                res += content
            else:
                if print_process:
                    # Print delta for non-incremental
                    delta = content[len(res):]
                    print(delta, end="", flush=True)
                res = content

        except (AttributeError, IndexError):
            continue

    if print_process:
        print("")  # New line at end

    return {"output": res}


# ============================================================================
# ArxivFetcher - arXiv paper fetcher
# ============================================================================
# Ref: https://github.com/yangjunxiao/paper-pulse
LOGGER_ARXIV = logging.getLogger(__name__)
ARXIV_RSS_API = "http://export.arxiv.org/api/query"
REQUEST_TIMEOUT_SECONDS = 60
REQUEST_MIN_INTERVAL_SECONDS = 12.0
DEFAULT_MAX_RETRIES = 5
RETRY_BACKOFF_FACTOR = 2.0
RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)
DEFAULT_HEADERS = {
    "User-Agent": "DeepInnovator/1.0",
    "Accept": "application/atom+xml; charset=utf-8",
    "Accept-Encoding": "gzip, deflate",
}


class ArxivFetcher:

    def __init__(
        self,
        page_size: int = 200,
        session: Optional[requests.Session] = None,
        categories: Optional[Sequence[str]] = None,
    ) -> None:
        self.page_size = page_size
        self.categories: tuple[str, ...] = self._normalize_categories(categories)
        self._category_clause: Optional[str] = self._build_category_clause(self.categories)
        self.session = session or self._build_session()
        self._last_request_ts: float = 0.0

    def fetch_by_id(self, arxiv_id: str) -> Optional[dict]:
        extracted_id = self._extract_arxiv_id(arxiv_id)
        if not extracted_id:
            LOGGER_ARXIV.warning("Failed to extract valid arXiv ID from input: %s", arxiv_id)
            return None
        
        LOGGER_ARXIV.info("Fetching arXiv paper with ID: %s", extracted_id)
        
        params = {
            "id_list": extracted_id,
            "max_results": 1,
        }
        
        feed = self._request_feed(params)
        if feed is None:
            LOGGER_ARXIV.warning("Failed to fetch arXiv paper with ID: %s", extracted_id)
            return None
        
        entries = getattr(feed, "entries", [])
        if not entries:
            LOGGER_ARXIV.warning("No paper found with arXiv ID: %s", extracted_id)
            return None
        
        try:
            paper = self._parse_entry(entries[0])
            LOGGER_ARXIV.info("Successfully fetched paper: %s", paper.get("title", ""))
            return paper
        except Exception as exc:
            LOGGER_ARXIV.error("Failed to parse arXiv entry for ID %s: %s", extracted_id, exc)
            return None

    def fetch_by_ids(self, arxiv_ids: Sequence[str]) -> List[dict]:
        extracted_ids = []
        for arxiv_id in arxiv_ids:
            extracted = self._extract_arxiv_id(arxiv_id)
            if extracted:
                extracted_ids.append(extracted)
            else:
                LOGGER_ARXIV.warning("Failed to extract valid arXiv ID from input: %s", arxiv_id)
        
        if not extracted_ids:
            LOGGER_ARXIV.warning("No valid arXiv IDs found")
            return []
        
        LOGGER_ARXIV.info("Fetching %d arXiv papers by IDs", len(extracted_ids))
        
        papers: List[dict] = []
        batch_size = 10
        
        for i in range(0, len(extracted_ids), batch_size):
            batch_ids = extracted_ids[i:i + batch_size]
            id_list_str = ",".join(batch_ids)
            
            params = {
                "id_list": id_list_str,
                "max_results": len(batch_ids),
            }
            
            feed = self._request_feed(params)
            if feed is None:
                LOGGER_ARXIV.warning("Failed to fetch arXiv batch: %s", id_list_str)
                continue
            
            entries = getattr(feed, "entries", [])
            for entry in entries:
                try:
                    paper = self._parse_entry(entry)
                    papers.append(paper)
                except Exception as exc:
                    LOGGER_ARXIV.warning("Failed to parse arXiv entry: %s", exc)
                    continue
        
        LOGGER_ARXIV.info("Successfully fetched %d papers out of %d requested", len(papers), len(extracted_ids))
        return papers

    def fetch_by_date(self, target_date: date, max_results: Optional[int] = None) -> List[dict]:
        start = target_date.strftime("%Y%m%d0000")
        end = target_date.strftime("%Y%m%d2359")
        date_clause = f"submittedDate:[{start} TO {end}]"
        if self._category_clause:
            query = f"({date_clause}) AND ({self._category_clause})"
        else:
            query = date_clause
        LOGGER_ARXIV.info("Fetching arXiv papers for date %s using query '%s'", target_date.isoformat(), query)
        papers = self._fetch_with_query(search_query=query, max_results=max_results)
        LOGGER_ARXIV.info("Fetched %d arXiv papers for date %s", len(papers), target_date.isoformat())
        return papers

    @staticmethod
    def _extract_arxiv_id(arxiv_input: str) -> Optional[str]:
        if not arxiv_input:
            return None
        
        arxiv_input = arxiv_input.strip()
        
        link_pattern = r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?'
        match = re.search(link_pattern, arxiv_input, re.IGNORECASE)
        if match:
            return match.group(1)
        
        id_pattern = r'^(\d{4}\.\d{4,5})(?:v\d+)?$'
        match = re.match(id_pattern, arxiv_input)
        if match:
            return match.group(1)
        
        return None

    def _fetch_with_query(self, search_query: str, max_results: Optional[int]) -> List[dict]:
        papers: List[dict] = []
        seen_ids: set[str] = set()
        start_index = 0
        consecutive_failures = 0
        max_batch_failures = 5
        
        while True:
            remaining = None if max_results is None else max_results - len(papers)
            if remaining is not None and remaining <= 0:
                break
            
            batch_size = self.page_size if remaining is None else min(self.page_size, remaining)
            LOGGER_ARXIV.debug(
                "Requesting arXiv batch: query='%s', start=%d, max_results=%s",
                search_query,
                start_index,
                batch_size,
            )
            
            params = {
                "search_query": search_query,
                "start": start_index,
                "max_results": batch_size,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            }
            
            feed = self._request_feed(params)
            if feed is None:
                consecutive_failures += 1
                LOGGER_ARXIV.warning(
                    "Failed to fetch arXiv batch (query='%s', start=%d, size=%d), "
                    "failure %d/%d. Skipping this batch.",
                    search_query,
                    start_index,
                    batch_size,
                    consecutive_failures,
                    max_batch_failures,
                )

                if consecutive_failures >= max_batch_failures:
                    LOGGER_ARXIV.warning(
                        "Stopping arXiv fetch due to repeated batch failures. query='%s', start=%d",
                        search_query,
                        start_index,
                    )
                    break

                start_index += batch_size
                continue
            
            entries = getattr(feed, "entries", [])
            if not entries:
                LOGGER_ARXIV.debug("No arXiv entries returned for query '%s' (start=%d).", search_query, start_index)
                break
            
            consecutive_failures = 0

            for entry in entries:
                try:
                    parsed = self._parse_entry(entry)
                except Exception as exc:
                    LOGGER_ARXIV.warning("Failed to parse arXiv entry for query %s: %s", search_query, exc)
                    continue
                
                paper_id = parsed.get("id", "")
                if paper_id in seen_ids:
                    LOGGER_ARXIV.debug("Skipping duplicate arXiv paper id=%s", paper_id)
                    continue
                
                seen_ids.add(paper_id)
                papers.append(parsed)
            
            start_index += len(entries)
            if len(entries) < batch_size:
                LOGGER_ARXIV.debug(
                    "Received final batch for query '%s': %d entries (requested %d).",
                    search_query,
                    len(entries),
                    batch_size,
                )
                break
        
        return papers

    @staticmethod
    def _parse_entry(entry: feedparser.FeedParserDict) -> dict:
        authors = [author.name for author in entry.get("authors", [])]
        
        affiliations = []
        for author in entry.get("authors", []):
            affiliation = None
            if isinstance(author, dict):
                affiliation = author.get("affiliation") or author.get("arxiv_affiliation")
            else:
                affiliation = getattr(author, "affiliation", None) or getattr(author, "arxiv_affiliation", None)
            if affiliation:
                text = str(affiliation).strip()
                if text and text not in affiliations:
                    affiliations.append(text)
        
        entry_affiliation = entry.get("arxiv_affiliation")
        if entry_affiliation:
            text = str(entry_affiliation).strip()
            if text and text not in affiliations:
                affiliations.append(text)
        
        categories = []
        for tag in entry.get("tags", []):
            term = getattr(tag, "term", None)
            if not term and isinstance(tag, dict):
                term = tag.get("term")
            if term:
                categories.append(str(term))
        
        published = ArxivFetcher._parse_datetime(entry.get("published", entry.get("updated")))
        pdf_url = ArxivFetcher._extract_pdf_url(entry)
        
        return {
            "id": entry["id"],
            "title": entry["title"],
            "summary": entry.get("summary", ""),
            "authors": authors,
            "link": entry["link"],
            "pdf_url": pdf_url,
            "published": published,
            "source": "arxiv",
            "categories": categories,
            "affiliations": affiliations,
        }

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime:
        if not value:
            return datetime.utcnow()
        return date_parser.parse(value)

    def _request_feed(self, params: dict) -> Optional[feedparser.FeedParserDict]:
        self._respect_rate_limit()
        try:
            response = self.session.get(
                ARXIV_RSS_API,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            self._last_request_ts = time.monotonic()
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER_ARXIV.warning("Failed to fetch arXiv feed: %s", exc)
            self._last_request_ts = time.monotonic()
            return None

        feed = feedparser.parse(response.text)

        status = getattr(feed, "status", response.status_code)
        if status != 200:
            LOGGER_ARXIV.warning("Unexpected arXiv response status: %s", status)
        if getattr(feed, "bozo", False):
            bozo_exception = getattr(feed, "bozo_exception", None)
            LOGGER_ARXIV.debug("arXiv feed parsing issue: %s", bozo_exception)
        return feed

    def _respect_rate_limit(self) -> None:
        if self._last_request_ts <= 0:
            return
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < REQUEST_MIN_INTERVAL_SECONDS:
            sleep_for = REQUEST_MIN_INTERVAL_SECONDS - elapsed
            time.sleep(max(0.0, sleep_for))

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=DEFAULT_MAX_RETRIES,
            read=DEFAULT_MAX_RETRIES,
            connect=DEFAULT_MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF_FACTOR,
            status_forcelist=RETRY_STATUS_FORCELIST,
            allowed_methods=frozenset({"GET"}),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        for key, value in DEFAULT_HEADERS.items():
            session.headers.setdefault(key, value)
        return session

    @staticmethod
    def _normalize_categories(categories: Optional[Sequence[str]]) -> tuple[str, ...]:
        DEFAULT_CATEGORIES = ("cs.CL", "cs.LG", "cs.AI")
        if not categories:
            return DEFAULT_CATEGORIES
        normalized = [str(category).strip() for category in categories if str(category).strip()]
        if not normalized:
            return DEFAULT_CATEGORIES
        unique = tuple(dict.fromkeys(normalized))
        return unique

    @staticmethod
    def _build_category_clause(categories: Optional[Sequence[str]]) -> Optional[str]:
        if not categories:
            return None
        clauses = [f"cat:{category}" for category in categories if category]
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return " OR ".join(clauses)

    @staticmethod
    def _extract_pdf_url(entry: feedparser.FeedParserDict) -> Optional[str]:
        links = entry.get("links", []) or []
        for link_info in links:
            href = getattr(link_info, "href", None)
            link_type = getattr(link_info, "type", None)
            title = getattr(link_info, "title", None)
            if isinstance(link_info, dict):
                href = href or link_info.get("href")
                link_type = link_type or link_info.get("type")
                title = title or link_info.get("title")
            if not href:
                continue
            if isinstance(link_type, str) and "pdf" in link_type.lower():
                return href
            if isinstance(title, str) and "pdf" in title.lower():
                return href
        fallback = entry.get("link") or entry.get("id")
        if isinstance(fallback, str) and "/abs/" in fallback:
            pdf_url = fallback.replace("/abs/", "/pdf/")
            if not pdf_url.endswith(".pdf"):
                pdf_url = f"{pdf_url}.pdf"
            return pdf_url
        return None


# ============================================================================
# OCR Functions
# ============================================================================

PROMPT_TICKET_EXTRACTION = """
You are a helpful assistant that can extract text from images.
You are given a page of a paper.
You need to extract all the text from the page.
"""


def _pdf_to_png_base64_list(pdf_path: Path, poppler_path: Optional[str] = None) -> list[str]:
    try:
        from pdf2image import convert_from_path
    except Exception as e:
        raise RuntimeError("pdf2image is required to convert PDF to images; pip install pdf2image, and ensure poppler is installed") from e

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError(f"PDF has no pages or read failed: {pdf_path}")

    result: list[str] = []
    for page_idx in range(1, total_pages + 1):
        images = convert_from_path(
            str(pdf_path),
            first_page=page_idx,
            last_page=page_idx,
            poppler_path=poppler_path,
        )
        if not images:
            continue
        buf = BytesIO()
        images[0].save(buf, format="PNG")
        png_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        result.append(f"data:image/png;base64,{png_base64}")

    if not result:
        raise ValueError(f"Failed to convert PDF to images: {pdf_path}")

    return result


def call_qwen_ocr(
    pdf_path: Path,
    prompt: str = PROMPT_TICKET_EXTRACTION,
    api_key: Optional[str] = None,
    poppler_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Use Qwen VL OCR to perform OCR recognition on PDF.
    
    Input a single pdf_path, returns {'data': {'ocr_text': ...}}
    Multi-page PDFs will be OCR'd page by page and concatenated.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    poppler_path = poppler_path or os.getenv("POPPLER_PATH")
    image_urls = _pdf_to_png_base64_list(pdf_path, poppler_path=poppler_path)

    key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("no DASHSCOPE_API_KEY or api_key")

    client = OpenAI(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    page_texts: list[str] = []
    total_usage = None
    success = 0

    for image_url in image_urls:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "min_pixels": 32 * 32 * 3,
                            "max_pixels": 32 * 32 * 8192,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        completion = client.chat.completions.create(
            model="qwen-vl-ocr-latest",
            messages=messages,
        )

        content = (
            completion.choices[0].message.content
            if completion.choices and completion.choices[0].message
            else ""
        )
        page_texts.append(content or "")
        if content:
            success += 1

        # Accumulate usage (may be None)
        if completion.usage:
            usage_dict = completion.usage.model_dump()
            if total_usage is None:
                total_usage = usage_dict
            else:
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if k in usage_dict and k in total_usage:
                        total_usage[k] = (total_usage.get(k) or 0) + (usage_dict.get(k) or 0)

    full_text = "\n\n".join(page_texts)

    return {
        "data": {
            "ocr_text": full_text,
            "num_pages": len(image_urls),
            "num_successful": success,
            "model": "qwen-vl-ocr-latest",
            "usage": total_usage,
        }
    }


# ============================================================================
# RawPaper - Paper download and OCR processing
# ============================================================================

ARXIV_API_URL = "http://export.arxiv.org/api/query"
REQUEST_MIN_INTERVAL_SECONDS_ARXIV = 3.0
_last_request_ts_arxiv: float = 0.0


def is_pdf_complete(file_path: Path) -> bool:
    """
    Check if a PDF file is complete by verifying file size and EOF marker.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        True if the file is complete, False otherwise
    """
    try:
        if file_path.stat().st_size == 0:
            logger.warning(f"File size is 0, download may be incomplete: {file_path}")
            return False
        
        with open(file_path, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            
            read_size = min(2048, file_size)
            f.seek(max(0, file_size - read_size))
            tail_content = f.read()
        
        if b'%%EOF' not in tail_content:
            logger.warning(f"File missing PDF EOF marker, download may be incomplete: {file_path}")
            return False
        
        return True
    except Exception as e:
        logger.warning(f"Error checking PDF completeness: {e}, treating as incomplete")
        return False


class RawPaper:
    
    def __init__(self, paper_dict: dict):
        self.id = paper_dict.get('id', '')
        self.title = paper_dict.get('title', '')
        self.summary = paper_dict.get('summary', '')
        self.authors = paper_dict.get('authors', [])
        self.link = paper_dict.get('link', '')
        self.pdf_url = paper_dict.get('pdf_url', '')
        self.published = paper_dict.get('published')
        self.source = paper_dict.get('source', 'arxiv')
        self.categories = paper_dict.get('categories', [])
        self.affiliations = paper_dict.get('affiliations', [])
    
    def extract_filename(self) -> Optional[str]:
        if not self.id:
            return None
        
        pattern = r'/(\d{4}\.\d{4,5}(?:v\d+)?)(?:/|$)'
        match = re.search(pattern, self.id)
        if match:
            arxiv_id = match.group(1)
            filename = arxiv_id.replace('.', '-')
            return filename
        return None
    
    def check_pdf_exists(self, save_path: Path, filename: Optional[str] = None) -> Optional[Path]:
        if filename is None:
            filename = self.extract_filename()
            if not filename:
                return None
        
        file_path = save_path / f"{filename}.pdf"
        
        if file_path.exists():
            return file_path
        
        return None
    
    def _is_pdf_complete(self, file_path: Path) -> bool:
        return is_pdf_complete(file_path)
    
    def download_pdf(self, save_path: Path, filename: Optional[str] = None) -> bool:
        if not self.pdf_url:
            logger.warning(f"Paper has no PDF URL: {self.id}")
            return False
        
        if filename is None:
            filename = self.extract_filename()
            if not filename:
                logger.warning(f"Failed to extract filename from ID: {self.id}")
                return False
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        file_path = save_path / f"{filename}.pdf"
        
        if file_path.exists():
            if self._is_pdf_complete(file_path):
                logger.info(f"File already exists and is complete (ID: {self.id}), skipping download: {file_path}")
                return True
            else:
                logger.warning(f"File exists but incomplete (ID: {self.id}), deleting and re-downloading: {file_path}")
                file_path.unlink()
        
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Downloading (attempt {attempt}/{max_retries}): {self.pdf_url}")
                response = requests.get(self.pdf_url, timeout=30, stream=True)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                    
            except Exception as e:
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except:
                        pass
                
                if attempt < max_retries:
                    wait_time = attempt * 2
                    logger.warning(f"Download failed (attempt {attempt}/{max_retries}): {e}, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to download PDF after {max_retries} retries {self.pdf_url}: {e}")
                    return False
        
        return False
    
    def extract_markdown_from_json(
        self,
        ocr_result: dict[str, Any],
        md_save_path: Path,
        filename: Optional[str] = None
    ) -> bool:
        if filename is None:
            filename = self.extract_filename()
            if not filename:
                logger.warning(f"Failed to extract filename from ID: {self.id}")
                return False
        
        try:
            data = ocr_result.get("data", {})
            ocr_text = data.get("ocr_text", "")
            if not isinstance(ocr_text, str):
                ocr_text = str(ocr_text)
            
            if not ocr_text:
                logger.warning(f"OCR result has no ocr_text field: {self.id}")
                return False
            
            md_save_path.mkdir(parents=True, exist_ok=True)
            
            md_path = md_save_path / f"{filename}.md"
            
            md_path.write_text(ocr_text, encoding="utf-8")
            logger.info(f"Markdown saved: {md_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract Markdown: {e}")
            return False
    
    def extract_ocr_content(
        self, 
        pdf_path: Path, 
        ocr_save_path: Path, 
        filename: Optional[str] = None,
        md_save_path: Optional[Path] = None,
        max_retries: int = 3,
        retry_delay: float = 10.0
    ) -> Optional[dict[str, Any]]:
        if filename is None:
            filename = self.extract_filename()
            if not filename:
                logger.warning(f"Failed to extract filename from ID: {self.id}")
                return None
        
        pdf_file_path = pdf_path / f"{filename}.pdf"
        
        if not pdf_file_path.exists():
            logger.warning(f"PDF file does not exist: {pdf_file_path}")
            return None
        
        ocr_save_path.mkdir(parents=True, exist_ok=True)
        ocr_output_path = ocr_save_path / f"{filename}.json"
        
        if ocr_output_path.exists():
            logger.info(f"OCR result already exists, skipping: {ocr_output_path}")
            try:
                with open(ocr_output_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                if md_save_path is not None:
                    md_path = md_save_path / f"{filename}.md"
                    if not md_path.exists():
                        logger.info(f"Markdown file does not exist, extracting from existing OCR result: {md_path}")
                        self.extract_markdown_from_json(result, md_save_path, filename)
                
                return result
            except Exception as e:
                logger.warning(f"Failed to read existing OCR result: {e}, will reprocess")
        
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    logger.info(f"Calling Qwen OCR (attempt {attempt}/{max_retries}): {pdf_file_path}")
                else:
                    logger.info(f"Calling Qwen OCR: {pdf_file_path}")
                
                result = call_qwen_ocr(pdf_file_path)
                ocr_output_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                logger.info(f"OCR result saved: {ocr_output_path}")
                
                if md_save_path is not None:
                    self.extract_markdown_from_json(result, md_save_path, filename)
                
                return result
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"OCR processing failed (attempt {attempt}/{max_retries}): {e}, "
                        f"retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"OCR processing failed after {max_retries} retries: {e}")
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"OCR processing error (attempt {attempt}/{max_retries}): {e}, "
                        f"retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"OCR processing failed after {max_retries} retries: {e}")
        
        logger.error(f"OCR processing finally failed {pdf_file_path} after {max_retries} attempts")
        return None
    
    def download_and_extract(
        self, 
        save_path: Path, 
        ocr_save_path: Path, 
        md_save_path: Optional[Path] = None,
        filename: Optional[str] = None, 
        auto_ocr: bool = True,
        max_retries: int = 3,
        retry_delay: float = 10.0
    ) -> bool:
        download_success = self.download_pdf(save_path, filename)
        if not download_success:
            return False
        
        if auto_ocr:
            ocr_result = self.extract_ocr_content(
                save_path, 
                ocr_save_path, 
                filename,
                md_save_path=md_save_path,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            return ocr_result is not None
        
        return True
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'title': self.title,
            'summary': self.summary,
            'authors': self.authors,
            'link': self.link,
            'pdf_url': self.pdf_url,
            'published': self.published.isoformat() if self.published else None,
            'source': self.source,
            'categories': self.categories,
            'affiliations': self.affiliations,
        }
    
    def __repr__(self) -> str:
        return f"RawPaper(id={self.id}, title={self.title[:50]}...)"


def download_arxiv_papers(arxiv_papers: list[str], DATAPATH: str, parallel: int = 5):
    save_path = Path(f'{DATAPATH}/raw_paper/original_paper')
    ocr_save_path = Path(f'{DATAPATH}/raw_paper/paper_json')
    md_save_path = Path(f'{DATAPATH}/raw_paper/paper_md')
    try:
        fetcher = ArxivFetcher()
        paper_dicts = fetcher.fetch_by_ids(arxiv_papers)
        max_workers = max(1, parallel or 1)

        def _download(paper_dict: dict):
            paper = RawPaper(paper_dict)
            filename = paper.extract_filename()
            if not filename:
                logger.warning(f"Failed to extract filename from ID: {paper.id}")
                return None

            print(f"\nDownloading paper: {paper.title[:60]}...")
            print(f"  ID: {paper.id}")
            print(f"  Filename: {filename}.pdf")
            ok = paper.download_pdf(save_path, filename)
            return (paper, filename) if ok else None

        downloaded: list[tuple[RawPaper, str]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_download, paper_dict) for paper_dict in paper_dicts]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        downloaded.append(result)
                except Exception as e:
                    logger.error(f"Error downloading paper: {e}", exc_info=True)

        def _ocr(paper_and_name: tuple[RawPaper, str]):
            paper, filename = paper_and_name
            print(f"\nOCR processing: {paper.title[:60]}...")
            paper.extract_ocr_content(
                save_path,
                ocr_save_path,
                filename,
                md_save_path=md_save_path,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_ocr, item) for item in downloaded]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing OCR for paper: {e}", exc_info=True)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        print(f"\nProcessing failed: {e}")


def get_arxiv_ids_by_title(title: str, max_results: int = 5) -> List[str]:
    """
    Search arXiv for papers by title and return matching arXiv IDs.
    
    Args:
        title: Paper title to search for
        max_results: Maximum number of results to return
    
    Returns:
        List of arXiv IDs (e.g., ["2503.00258", "2401.12345"])
    """
    if not title or not title.strip():
        return []
    
    fetcher = ArxivFetcher(page_size=min(max_results, 50))
    # Use title search query
    query = f'ti:"{title}"'
    
    papers = fetcher._fetch_with_query(search_query=query, max_results=max_results)  # type: ignore[attr-defined]
    
    arxiv_ids = []
    for paper in papers:
        arxiv_id = fetcher._extract_arxiv_id(paper.get('id', ''))
        if arxiv_id:
            arxiv_ids.append(arxiv_id)
    
    return arxiv_ids


def extract_paper_references(paper_md_path: str | Path, config=None) -> dict:
    """
    Extract all references from a paper's Reference section.
    
    Args:
        paper_md_path: Path to the paper markdown file
        config: Optional config dictionary. If None, will load config using load_config()
        
    Returns:
        Dictionary containing extracted references with structure:
        {
            'references': [
                {
                    'title': str,
                    'authors': List[str],
                    'url': str,
                    ...
                },
                ...
            ]
        }
    """
    paper_md_path = Path(paper_md_path)
    if not paper_md_path.exists():
        raise FileNotFoundError(f"Paper file not found: {paper_md_path}")
    
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Read paper content
    with open(paper_md_path, 'r', encoding='utf-8') as f:
        paper_content = f.read()
    
    # Get agent configuration
    agent_config = config['agents']['general_paper_refence']
    
    # Format prompt with paper content
    prompt = agent_config['prompt'].format(
        paper_content=paper_content,
        schema=json.dumps(
            OmegaConf.to_container(agent_config['schema'], resolve=True),
            indent=2,
            ensure_ascii=False
        )
    )
    
    # Call agent to extract references
    result = call_agent('general_paper_refence', prompt, config)
    if not result:
        return {'references': []}
    
    references = result.get('references', [])
    
    # Fill in missing URLs by searching arXiv
    if references:
        filled_count = 0
        
        for ref in references:
            title = ref.get('title', 'N/A')
            url = ref.get('url', '').strip()
            
            # If URL is empty, try to find it on arXiv
            if not url and title and title != 'N/A':
                arxiv_ids = get_arxiv_ids_by_title(title, max_results=1)
                
                if arxiv_ids:
                    arxiv_id = arxiv_ids[0]
                    url = f"https://arxiv.org/abs/{arxiv_id}"
                    ref['url'] = url
                    filled_count += 1
    
    return result
