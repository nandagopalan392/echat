"""
Ollama model scraper to fetch available models from Ollama library
with comprehensive fallback list and real-time model discovery
"""

import requests
import time
import logging
import re
import json
from typing import Dict, List, Any
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Comprehensive fallback list with latest Ollama models (updated based on ollama.com/search)
FALLBACK_MODELS = {
    # Latest Large Language Models from Ollama Search
    "deepseek-r1": {
        "name": "deepseek-r1",
        "category": "llm",
        "description": "DeepSeek-R1 is a family of open reasoning models with performance approaching that of leading models",
        "tags": ["deepseek", "reasoning", "latest"],
        "size": "Various sizes: 1.5b, 7b, 8b, 14b, 32b, 70b, 671b",
        "source": "ollama",
        "variants": ["1.5b", "7b", "8b", "14b", "32b", "70b", "671b"]
    },
    "llama4": {
        "name": "llama4",
        "category": "llm",
        "description": "Meta's latest collection of multimodal models",
        "tags": ["meta", "multimodal", "vision", "tools", "latest"],
        "size": "Various sizes: 16x17b, 128x17b",
        "source": "ollama",
        "variants": ["16x17b", "128x17b"]
    },
    "qwen3": {
        "name": "qwen3", 
        "category": "llm",
        "description": "Qwen3 is the latest generation of large language models in Qwen series, offering dense and MoE models",
        "tags": ["alibaba", "tools", "thinking", "latest"],
        "size": "Various sizes: 0.6b, 1.7b, 4b, 8b, 14b, 30b, 32b, 235b",
        "source": "ollama",
        "variants": ["0.6b", "1.7b", "4b", "8b", "14b", "30b", "32b", "235b"]
    },
    "gemma3": {
        "name": "gemma3",
        "category": "llm",
        "description": "The current, most capable model that runs on a single GPU",
        "tags": ["google", "vision", "latest"],
        "size": "Various sizes: 1b, 4b, 12b, 27b",
        "source": "ollama",
        "variants": ["1b", "4b", "12b", "27b"]
    },
    "gemma3n": {
        "name": "gemma3n",
        "category": "llm", 
        "description": "Gemma 3n models are designed for efficient execution on everyday devices",
        "tags": ["google", "efficient", "mobile"],
        "size": "Various sizes: e2b, e4b",
        "source": "ollama",
        "variants": ["e2b", "e4b"]
    },
    "phi4": {
        "name": "phi4",
        "category": "llm",
        "description": "Phi-4 is a 14B parameter, state-of-the-art open model from Microsoft",
        "tags": ["microsoft", "14b"],
        "size": "14b",
        "source": "ollama"
    },
    "phi4-mini": {
        "name": "phi4-mini",
        "category": "llm",
        "description": "Phi-4-mini brings significant enhancements in multilingual support, reasoning, and mathematics",
        "tags": ["microsoft", "3.8b", "tools"],
        "size": "3.8b",
        "source": "ollama"
    },
    "llama3.3": {
        "name": "llama3.3",
        "category": "llm",
        "description": "New state of the art 70B model. Llama 3.3 70B offers similar performance compared to the Llama 3.1 405B model",
        "tags": ["meta", "70b", "tools"],
        "size": "70b",
        "source": "ollama"
    },
    "llama3.2": {
        "name": "llama3.2",
        "category": "llm",
        "description": "Meta's Llama 3.2 goes small with 1B and 3B models",
        "tags": ["meta", "tools", "small"],
        "size": "Various sizes: 1b, 3b",
        "source": "ollama",
        "variants": ["1b", "3b"]
    },
    "llama3.2-vision": {
        "name": "llama3.2-vision",
        "category": "vision",
        "description": "Llama 3.2 Vision is a collection of instruction-tuned image reasoning generative models",
        "tags": ["meta", "vision", "multimodal"],
        "size": "Various sizes: 11b, 90b",
        "source": "ollama",
        "variants": ["11b", "90b"]
    },
    "llama3.1": {
        "name": "llama3.1",
        "category": "llm",
        "description": "Llama 3.1 is a new state-of-the-art model from Meta available in 8B, 70B and 405B parameter sizes",
        "tags": ["meta", "tools"],
        "size": "Various sizes: 8b, 70b, 405b",
        "source": "ollama",
        "variants": ["8b", "70b", "405b"]
    },
    "llama3": {
        "name": "llama3",
        "category": "llm",
        "description": "Meta Llama 3: The most capable openly available LLM to date",
        "tags": ["meta", "8b", "70b"],
        "size": "Various sizes: 8b, 70b",
        "source": "ollama",
        "variants": ["8b", "70b"]
    },
    "qwen2.5": {
        "name": "qwen2.5",
        "category": "llm",
        "description": "Qwen2.5 models are pretrained on Alibaba's latest large-scale dataset, encompassing up to 18 trillion tokens",
        "tags": ["alibaba", "tools", "multilingual"],
        "size": "Various sizes: 0.5b, 1.5b, 3b, 7b, 14b, 32b, 72b",
        "source": "ollama",
        "variants": ["0.5b", "1.5b", "3b", "7b", "14b", "32b", "72b"]
    },
    "qwen2.5-coder": {
        "name": "qwen2.5-coder",
        "category": "llm",
        "description": "The latest series of Code-Specific Qwen models, with significant improvements in code generation",
        "tags": ["alibaba", "code", "tools"],
        "size": "Various sizes: 0.5b, 1.5b, 3b, 7b, 14b, 32b",
        "source": "ollama",
        "variants": ["0.5b", "1.5b", "3b", "7b", "14b", "32b"]
    },
    "qwen2.5vl": {
        "name": "qwen2.5vl",
        "category": "vision",
        "description": "Flagship vision-language model of Qwen and also a significant leap from the previous Qwen2-VL",
        "tags": ["alibaba", "vision", "multimodal"],
        "size": "Various sizes: 3b, 7b, 32b, 72b",
        "source": "ollama",
        "variants": ["3b", "7b", "32b", "72b"]
    },
    "mistral": {
        "name": "mistral",
        "category": "llm",
        "description": "The 7B model released by Mistral AI, updated to version 0.3",
        "tags": ["mistral", "7b", "tools"],
        "size": "7b",
        "source": "ollama"
    },
    "mistral-small3.2": {
        "name": "mistral-small3.2",
        "category": "llm",
        "description": "An update to Mistral Small that improves on function calling, instruction following",
        "tags": ["mistral", "vision", "tools"],
        "size": "24b",
        "source": "ollama"
    },
    "mixtral": {
        "name": "mixtral",
        "category": "llm",
        "description": "A set of Mixture of Experts (MoE) model with open weights by Mistral AI",
        "tags": ["mistral", "moe", "tools"],
        "size": "Various sizes: 8x7b, 8x22b",
        "source": "ollama",
        "variants": ["8x7b", "8x22b"]
    },
    "devstral": {
        "name": "devstral",
        "category": "llm",
        "description": "Devstral: the best open source model for coding agents",
        "tags": ["mistral", "code", "tools"],
        "size": "24b",
        "source": "ollama"
    },
    "gemma2": {
        "name": "gemma2",
        "category": "llm",
        "description": "Google Gemma 2 is a high-performing and efficient model",
        "tags": ["google"],
        "size": "Various sizes: 2b, 9b, 27b",
        "source": "ollama",
        "variants": ["2b", "9b", "27b"]
    },
    "phi3": {
        "name": "phi3",
        "category": "llm",
        "description": "Phi-3 is a family of lightweight 3B (Mini) and 14B (Medium) state-of-the-art open models by Microsoft",
        "tags": ["microsoft"],
        "size": "Various sizes: 3.8b, 14b",
        "source": "ollama",
        "variants": ["3.8b", "14b"]
    },
    "deepseek-v3": {
        "name": "deepseek-v3",
        "category": "llm",
        "description": "A strong Mixture-of-Experts (MoE) language model with 671B total parameters",
        "tags": ["deepseek", "moe", "large"],
        "size": "671b",
        "source": "ollama"
    },
    "codellama": {
        "name": "codellama",
        "category": "llm",
        "description": "A large language model that can use text prompts to generate and discuss code",
        "tags": ["meta", "code"],
        "size": "Various sizes: 7b, 13b, 34b, 70b",
        "source": "ollama",
        "variants": ["7b", "13b", "34b", "70b"]
    },
    "granite3.3": {
        "name": "granite3.3",
        "category": "llm",
        "description": "IBM Granite 2B and 8B models are 128K context length language models",
        "tags": ["ibm", "tools"],
        "size": "Various sizes: 2b, 8b",
        "source": "ollama",
        "variants": ["2b", "8b"]
    },
    "llava": {
        "name": "llava",
        "category": "vision",
        "description": "LLaVA is a novel end-to-end trained large multimodal model",
        "tags": ["vision", "multimodal"],
        "size": "Various sizes: 7b, 13b, 34b",
        "source": "ollama",
        "variants": ["7b", "13b", "34b"]
    },
    "smollm2": {
        "name": "smollm2",
        "category": "llm",
        "description": "SmolLM2 is a family of compact language models",
        "tags": ["small", "tools"],
        "size": "Various sizes: 135m, 360m, 1.7b",
        "source": "ollama",
        "variants": ["135m", "360m", "1.7b"]
    },
    "starcoder2": {
        "name": "starcoder2",
        "category": "llm",
        "description": "StarCoder2 is the next generation of transparently trained open code LLMs",
        "tags": ["code"],
        "size": "Various sizes: 3b, 7b, 15b",
        "source": "ollama",
        "variants": ["3b", "7b", "15b"]
    },
    "deepseek-coder-v2": {
        "name": "deepseek-coder-v2",
        "category": "llm",
        "description": "An open-source Mixture-of-Experts code language model",
        "tags": ["deepseek", "code", "moe"],
        "size": "Various sizes: 16b, 236b",
        "source": "ollama",
        "variants": ["16b", "236b"]
    },
    "openthinker": {
        "name": "openthinker",
        "category": "llm",
        "description": "A fully open-source family of reasoning models built using a dataset derived by distilling DeepSeek-R1",
        "tags": ["reasoning", "opensource"],
        "size": "Various sizes: 7b, 32b",
        "source": "ollama",
        "variants": ["7b", "32b"]
    },
    "magistral": {
        "name": "magistral",
        "category": "llm",
        "description": "Magistral is a small, efficient reasoning model with 24B parameters",
        "tags": ["reasoning", "tools", "thinking"],
        "size": "24b",
        "source": "ollama"
    },
    
    # Embedding Models
    "nomic-embed-text": {
        "name": "nomic-embed-text",
        "category": "embedding",
        "description": "A high-performing open embedding model with a large token context window",
        "tags": ["nomic", "embedding", "text"],
        "size": "274MB",
        "source": "ollama"
    },
    "mxbai-embed-large": {
        "name": "mxbai-embed-large",
        "category": "embedding",
        "description": "State-of-the-art large embedding model from mixedbread.ai",
        "tags": ["mixedbread", "large", "embedding"],
        "size": "335m",
        "source": "ollama"
    },
    "snowflake-arctic-embed": {
        "name": "snowflake-arctic-embed",
        "category": "embedding",
        "description": "A suite of text embedding models by Snowflake, optimized for performance",
        "tags": ["snowflake", "arctic", "embedding"],
        "size": "Various sizes: 22m, 33m, 110m, 137m, 335m",
        "source": "ollama",
        "variants": ["22m", "33m", "110m", "137m", "335m"]
    },
    "snowflake-arctic-embed2": {
        "name": "snowflake-arctic-embed2",
        "category": "embedding",
        "description": "Snowflake's frontier embedding model. Arctic Embed 2.0 adds multilingual support",
        "tags": ["snowflake", "multilingual", "embedding"],
        "size": "568m",
        "source": "ollama"
    },
    "all-minilm": {
        "name": "all-minilm",
        "category": "embedding",
        "description": "Embedding models on very large sentence level datasets",
        "tags": ["sentence-transformers", "minilm"],
        "size": "Various sizes: 22m, 33m",
        "source": "ollama",
        "variants": ["22m", "33m"]
    },
    "bge-large": {
        "name": "bge-large",
        "category": "embedding",
        "description": "Embedding model from BAAI mapping texts to vectors",
        "tags": ["bge", "large", "embedding"],
        "size": "335m",
        "source": "ollama"
    },
    "bge-m3": {
        "name": "bge-m3",
        "category": "embedding",
        "description": "BGE-M3 is a new model from BAAI distinguished for its versatility",
        "tags": ["bge", "multilingual", "embedding"],
        "size": "567m",
        "source": "ollama"
    },
    "granite-embedding": {
        "name": "granite-embedding",
        "category": "embedding",
        "description": "The IBM Granite Embedding models are text-only dense biencoder embedding models",
        "tags": ["ibm", "embedding"],
        "size": "Various sizes: 30m, 278m",
        "source": "ollama",
        "variants": ["30m", "278m"]
    }
}


def scrape_ollama_search_page() -> Dict[str, Any]:
    """
    Scrape the Ollama search page to get comprehensive model list with sizes
    """
    try:
        logger.info("Scraping Ollama search page...")
        
        url = "https://ollama.com/search"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            models = {}
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find model links - they typically have format /library/model-name
            model_links = soup.find_all('a', href=re.compile(r'/library/[\w\-\.]+'))
            
            for link in model_links:
                href = link.get('href', '')
                if '/library/' in href:
                    model_name = href.split('/library/')[-1]
                    
                    # Extract model info from the link text and surrounding elements
                    link_text = link.get_text(strip=True)
                    parent = link.parent
                    
                    # Try to find size information in the parent elements
                    size_info = "Unknown"
                    description = f"{model_name} model from Ollama library"
                    
                    if parent:
                        parent_text = parent.get_text()
                        # Look for size patterns like "7b", "13b", "70b", "1.5b", etc.
                        size_matches = re.findall(r'\b\d+\.?\d*[bBmMgGtT]\b', parent_text)
                        if size_matches:
                            size_info = ', '.join(size_matches)
                        
                        # Look for pull counts and description
                        if 'Pulls' in parent_text:
                            description_match = re.search(r'([^.]+\.)', parent_text)
                            if description_match:
                                description = description_match.group(1).strip()
                    
                    # Categorize the model
                    category = categorize_model(model_name)
                    
                    # Determine available variants from size info
                    variants = []
                    if size_info != "Unknown":
                        variants = [s.strip() for s in size_info.split(',')]
                    
                    models[model_name] = {
                        "name": model_name,
                        "category": category,
                        "description": description,
                        "tags": [category],
                        "size": size_info,
                        "source": "ollama"
                    }
                    
                    if variants:
                        models[model_name]["variants"] = variants
            
            if models:
                logger.info(f"Successfully scraped {len(models)} models from search page")
                return models
                
    except Exception as e:
        logger.warning(f"Failed to scrape search page: {e}")
    
    return {}


def fetch_model_details(model_name: str) -> Dict[str, Any]:
    """
    Fetch detailed information for a specific model from its library page
    """
    try:
        url = f"https://ollama.com/library/{model_name}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract description
            description = f"{model_name} model from Ollama library"
            desc_elem = soup.find('meta', {'name': 'description'})
            if desc_elem and desc_elem.get('content'):
                description = desc_elem.get('content').strip()
            
            # Extract available variants and sizes
            variants = []
            sizes = []
            
            # Look for tag information
            tag_elements = soup.find_all(string=re.compile(r'\b\d+\.?\d*[bBmMgGtT]\b'))
            for tag in tag_elements:
                if isinstance(tag, str):
                    size_matches = re.findall(r'\b(\d+\.?\d*[bBmMgGtT])\b', tag)
                    variants.extend(size_matches)
            
            # Remove duplicates and sort
            variants = sorted(list(set(variants)))
            
            return {
                "name": model_name,
                "category": categorize_model(model_name),
                "description": description,
                "tags": [categorize_model(model_name)],
                "size": ', '.join(variants) if variants else "Unknown",
                "source": "ollama",
                "variants": variants if variants else []
            }
            
    except Exception as e:
        logger.warning(f"Failed to fetch details for {model_name}: {e}")
    
    return None


def scrape_ollama_registry() -> Dict[str, Any]:
    """
    Try to scrape from Ollama API endpoints
    """
    try:
        logger.info("Attempting to fetch from Ollama API...")
        
        # Try the tags API endpoint  
        api_url = "https://ollama.com/api/tags"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            models = {}
            
            if 'models' in data:
                for model in data['models'][:100]:  # Limit to first 100
                    name = model.get('name', '').split(':')[0]  # Remove tag part
                    if name:
                        # Convert size from bytes to human readable
                        size_bytes = model.get('size', 0)
                        if size_bytes > 0:
                            if size_bytes >= 1073741824:  # 1GB
                                size = f"{size_bytes / 1073741824:.1f}GB"
                            elif size_bytes >= 1048576:  # 1MB
                                size = f"{size_bytes / 1048576:.0f}MB"
                            else:
                                size = f"{size_bytes}B"
                        else:
                            size = "Unknown"
                        
                        category = categorize_model(name)
                        models[name] = {
                            "name": name,
                            "category": category,
                            "description": f"{name} model from Ollama",
                            "tags": [category],
                            "size": size,
                            "source": "ollama"
                        }
                
                logger.info(f"Successfully fetched {len(models)} models from API")
                return models
                
    except Exception as e:
        logger.warning(f"Failed to fetch from API: {e}")
    
    return {}


def scrape_ollama_library() -> Dict[str, Any]:
    """
    Scrape Ollama library website to get available models
    """
    try:
        logger.info("Scraping Ollama library website...")
        
        url = "https://ollama.com/library"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            models = {}
            # Simple text parsing to extract model names
            content = response.text.lower()
            
            # Look for common model patterns in the HTML
            model_patterns = [
                'llama3', 'llama2', 'codellama', 'mistral', 'mixtral',
                'gemma', 'phi3', 'qwen', 'deepseek-coder', 'solar',
                'neural-chat', 'starling-lm', 'openchat', 'zephyr',
                'orca-mini', 'vicuna', 'wizardlm', 'tinyllama',
                'dolphin-mistral', 'llava', 'nomic-embed-text'
            ]
            
            for pattern in model_patterns:
                if pattern in content:
                    category = categorize_model(pattern)
                    models[pattern] = {
                        "name": pattern,
                        "category": category,
                        "description": f"{pattern} model from Ollama library",
                        "tags": [category],
                        "size": "Unknown",
                        "source": "ollama"
                    }
            
            if models:
                logger.info(f"Successfully scraped {len(models)} models from library")
                return models
        
    except Exception as e:
        logger.warning(f"Failed to scrape library website: {e}")
    
    return {}


def categorize_model(model_name: str) -> str:
    """
    Categorize a model based on its name
    """
    name_lower = model_name.lower()
    
    if any(keyword in name_lower for keyword in ['embed', 'embedding']):
        return 'embedding'
    elif any(keyword in name_lower for keyword in ['llava', 'vision', 'multimodal']):
        return 'vision'
    elif any(keyword in name_lower for keyword in ['rerank']):
        return 'reranker'
    else:
        return 'llm'


def get_available_models(use_cache: bool = True) -> Dict[str, Any]:
    """
    Get all available models from Ollama, with comprehensive scraping and fallback
    """
    models = {}
    
    # Try to fetch from various sources in order of preference
    try:
        # First try the API endpoint for real-time data
        api_models = scrape_ollama_registry()
        if api_models:
            models.update(api_models)
            logger.info(f"Got {len(api_models)} models from API")
        
        # Then try scraping the search page
        search_models = scrape_ollama_search_page()
        if search_models:
            # Merge with preference for search page data (more detailed)
            for name, model_info in search_models.items():
                if name in models:
                    # Update existing entry with more detailed info from search page
                    models[name].update(model_info)
                else:
                    models[name] = model_info
            logger.info(f"Got {len(search_models)} models from search page")
        
        # Try library website as additional source
        library_models = scrape_ollama_library()
        if library_models:
            for name, model_info in library_models.items():
                if name not in models:
                    models[name] = model_info
            logger.info(f"Got {len(library_models)} additional models from library")
        
    except Exception as e:
        logger.warning(f"Error fetching models from online sources: {e}")
    
    # Always merge with fallback models to ensure comprehensive coverage
    final_models = FALLBACK_MODELS.copy()
    
    # Update fallback models with any better data from online sources
    for name, model_info in models.items():
        if name in final_models:
            # Update existing fallback entry with online data
            final_models[name].update(model_info)
        else:
            # Add new model discovered online
            final_models[name] = model_info
    
    # Expand models that have variants
    expanded_models = {}
    for name, model_info in final_models.items():
        expanded_models[name] = model_info.copy()
        
        # If model has variants, create separate entries for each variant
        if 'variants' in model_info and model_info['variants']:
            for variant in model_info['variants']:
                variant_name = f"{name}:{variant}"
                variant_info = model_info.copy()
                variant_info['name'] = variant_name
                variant_info['size'] = variant
                # Remove variants list from individual variant entries
                if 'variants' in variant_info:
                    del variant_info['variants']
                expanded_models[variant_name] = variant_info
    
    logger.info(f"Returning {len(expanded_models)} total models (including variants)")
    return expanded_models


def get_available_ollama_models(use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Get available models as a list for backward compatibility with main.py
    """
    models_dict = get_available_models(use_cache)
    # Convert dictionary to list format expected by main.py
    models_list = []
    for model_name, model_info in models_dict.items():
        models_list.append({
            'name': model_name,
            'category': model_info.get('category', 'llm'),
            'description': model_info.get('description', ''),
            'tags': model_info.get('tags', []),
            'size': model_info.get('size', 'Unknown'),
            'source': model_info.get('source', 'ollama')
        })
    return models_list


if __name__ == "__main__":
    # Test the scraper
    logging.basicConfig(level=logging.INFO)
    models = get_available_models()
    print(f"Found {len(models)} models:")
    for name, info in list(models.items())[:10]:
        print(f"  {name}: {info['category']} - {info['description']}")
