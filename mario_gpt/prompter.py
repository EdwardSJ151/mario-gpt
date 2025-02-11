from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.interpolate import splprep, splev
import torch
from scipy import stats
from transformers import pipeline

# from mario_gpt.dataset import MarioDataset
from mario_gpt.utils import view_level

STATISTICS = {
    "special enemy": np.array([1.0, 2.0, 3.0]),
    "pipe": np.array([1.0, 2.0, 5.0]),
    "ground block": np.array([24.0, 87.0, 150.0]),

    "hard block": np.array([1.0, 11.0, 23.0]),
    "coin block": np.array([1.0, 2.0, 6.0]),
    "breakable block": np.array([1.0, 19.0, 38.0]),
    "coin": np.array([1.0, 5.0, 10.0]),
    "powerup": np.array([1.0, 2.0, 3.0]),
    "goomba": np.array([1.0, 2.0, 5.0]),
    "koopa": np.array([1.0, 3.0, 6.0]),
}

FEATURE_EXTRACTION_MODEL = "facebook/bart-base"

class Prompter:
    def __init__(
        self,
        level_tokenizer,
        prompter_model: str = FEATURE_EXTRACTION_MODEL,
        use_raw_counts: bool = False,
        statistics: Optional[Dict[str, Any]] = None,
    ):
        self.prompter_model = prompter_model
        self.feature_extraction = pipeline(
            "feature-extraction",
            model=prompter_model,
            tokenizer=prompter_model,
            framework="pt",
        )

        self.level_tokenizer = level_tokenizer

        self.use_raw_counts = use_raw_counts
        self.statistics = statistics
        if statistics is None:
            self.statistics = STATISTICS

        self.entity_chars = {
            "pipe": ["<>", "()"],
            "special enemy": ["E", "B", "y"],
            "ground block": ["X"], # Ground blocks
            "hard block": ["#"], # Hard blocks
            "coin block": ["Q", "!", "2", "C"], # Interactible Blocks -> Question Block (Empty), Invisible Coin Block, Coin Brick Block
            "breakable block": ["S"], # Breakable blocks
            "koopa": ["r", "R", "k", "K"],
            "goomba": ["g", "G"],
            "powerup": ["1", "?", "U", "L"],
            "coin": ["o"],
        }

    def get_thresholds(self, entity_type: str) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics[entity_type]
        keywords = ["no", "little", "some", "many"]
        if entity_type == "ground block":
            keywords = ["little", "little", "some", "many"]
        return thresholds, keywords

    def count_entities(self, flattened_level: str, entity_type: str) -> int:
        if entity_type not in self.entity_chars:
            raise ValueError(f"Unknown entity type: {entity_type}")
            
        chars = self.entity_chars[entity_type]
        return sum(flattened_level.count(char) for char in chars)
    def count_entities(self, flattened_level: str, entity_type: str) -> int:
        if entity_type not in self.entity_chars:
            raise ValueError(f"Unknown entity type: {entity_type}")
            
        chars = self.entity_chars[entity_type]
        return sum(flattened_level.count(char) for char in chars)

    def generate_prompt(self, entity_type: str, flattened_level: str, level: str = None) -> Tuple[str, str]:
        if entity_type == "elevation":
            return self.elevation_prompt(flattened_level, level)
            
        count = self.count_entities(flattened_level, entity_type)
    def generate_prompt(self, entity_type: str, flattened_level: str, level: str = None) -> Tuple[str, str]:
        if entity_type == "elevation":
            return self.elevation_prompt(flattened_level, level)
            
        count = self.count_entities(flattened_level, entity_type)
        keyword = f"{count}"
        
        
        if not self.use_raw_counts:
            thresholds, keywords = self.get_thresholds(entity_type)
            threshold = np.digitize(count, thresholds, right=False)
            keyword = keywords[threshold]
            
        # Handle special plural cases
        plural = "special enemies" if entity_type == "special enemy" else f"{entity_type}s"
        return f"{keyword} {plural}", keyword
            

    def elevation_prompt(self, flattened_level: str, level: str):
        top_levels = level[:6]
        top_levels = level[:6]
        for t in top_levels:
            if "X" in t or "<" in t or ">" in t:
                return "high elevation", "high"
        return "low elevation", "low"

    def _flatten_level(self, string_level: List[str]) -> str:
        return "".join(string_level)

    def _flatten_level(self, string_level: List[str]) -> str:
        return "".join(string_level)

    def output_hidden(self, prompt: str, device: torch.device = torch.device("cpu")):
        return (
            self.feature_extraction(prompt, return_tensors="pt")[0]
            .mean(0)
            .to(device)
            .view(1, -1)
        )

    def dataset_statistics(self, dataset):
        counts = {
            "special enemy": [],
            "pipe": [],
            "ground block": [],
            "hard block": [],
            "coin block": [],
            "breakable block": [],
            "goomba": [],
            "koopa": [],
            "coin": [],
            "powerup": []
        }
        
        for i in range(len(dataset)):
            level, _ = dataset[i]
            str_level = self._flatten_level(view_level(level, dataset.tokenizer))

            # Count each entity type
            for entity_type in counts.keys():
                count = self.count_entities(str_level, entity_type)
                counts[entity_type].append(count)

        # Calculate statistics for each entity type
        d = {k: {} for k in counts.keys()}
        for entity_type, entity_counts in counts.items():
            d[entity_type] = stats.mstats.mquantiles(entity_counts, [0.33, 0.66, 0.95])

        return d

    def __call__(
        self, level: torch.Tensor = None, level_info: Optional[dict] = None, sample_prompt: bool = False
    ) -> Union[str, torch.Tensor]:
        device: torch.device = torch.device("cpu")
        prompt_dict = {}
                
        if not sample_prompt:
            if level is None:
                raise ValueError("Level must be provided if sample_prompt is not true!")
            str_level = view_level(level, self.level_tokenizer)
            flattened_level = self._flatten_level(str_level)
            device = level.device

            # Generate prompts for all entity types
            for entity_type in self.entity_chars.keys():
                count = sum(flattened_level.count(char) for char in self.entity_chars[entity_type])
                prompt_dict[entity_type] = f"{count} {entity_type}s"

            # Handle elevation separately
            elevation_prompt, elevation_keyword = self.elevation_prompt(flattened_level, str_level)
            prompt_dict["elevation_prompt"] = elevation_prompt

            # Generate prompts for all entity types
            for entity_type in self.entity_chars.keys():
                prompt, keyword = self.generate_prompt(entity_type, flattened_level, str_level)
                prompt_dict[entity_type] = prompt

            # Handle elevation separately
            elevation_prompt, elevation_keyword = self.elevation_prompt(flattened_level, str_level)
            prompt_dict["elevation_prompt"] = elevation_prompt
        else:
            str_level = None
            # Generate random prompts for all entity types
            for entity_type in self.entity_chars.keys():
                keywords = ["no", "little", "some", "many"]
                if entity_type == "ground block":
                    keywords = ["little", "little", "some", "many"]
                keyword = random.choice(keywords)
                prompt_dict[entity_type] = f"{keyword} {entity_type}s"

            # Handle elevation separately
            elevation_keyword = random.choice(["low", "high"])
            prompt_dict["elevation_prompt"] = f"{elevation_keyword} elevation"

        prompt = ", ".join(prompt_dict.values())
        hidden = self.output_hidden(prompt, device=device)
        return prompt, hidden, prompt_dict, str_level

