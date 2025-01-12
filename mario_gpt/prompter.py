from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats
from transformers import pipeline

from mario_gpt.dataset import MarioDataset
from mario_gpt.utils import view_level

STATISTICS = {
    "enemy": np.array([1.0, 4.0, 9.0]),
    "pipe": np.array([0.0, 3.0, 7.0]),
    "block": np.array([50.0, 75.0, 176.0]),

    "coin": np.array([1.0, 20.0, 50.0]),
    "powerup": np.array([1.0, 4.0, 9.0]),
    "goomba": np.array([1.0, 3.0, 7.0]),
    "koopa": np.array([1.0, 3.0, 7.0]),
}

FEATURE_EXTRACTION_MODEL = "facebook/bart-base"

# TODO: If this actually becomes a paper, have a "class_names" dict instad of making functions for each class
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

    @property
    def pipe_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["pipe"]
        keywords = ["no", "little", "some", "many"]
        return thresholds, keywords

    @property
    def enemy_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["enemy"]
        keywords = ["no", "little", "some", "many"]
        return thresholds, keywords

    @property
    def block_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["block"]
        keywords = ["little", "little", "some", "many"]
        return thresholds, keywords

    @property
    def coin_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["coin"]
        keywords = ["no", "little", "some", "many"]
        return thresholds, keywords

    @property
    def powerup_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["powerup"]
        keywords = ["no", "little", "some", "many"]
        return thresholds, keywords

    @property
    def goomba_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["goomba"]
        keywords = ["no", "little", "some", "many"]
        return thresholds, keywords

    @property
    def koopa_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["koopa"]
        keywords = ["no", "little", "some", "many"]
        return thresholds, keywords

    def count_pipes(self, flattened_level: str) -> int:
        return flattened_level.count("<>") + flattened_level.count("()")

    def count_enemies(self, flattened_level: str) -> int:
        return np.sum([flattened_level.count(char) for char in ["E", "B", "y"]])

    def count_blocks(self, flattened_level: str) -> int:
        return np.sum([flattened_level.count(char) for char in ["X", "S", "Q", "!", "2", "C", "#"]])

    def count_koopas(self, flattened_level: str) -> int:
        return np.sum([flattened_level.count(char) for char in ["r", "R", "k", "K"]])

    def count_goombas(self, flattened_level: str) -> int:
        return flattened_level.count("g") + flattened_level.count("G")

    def count_powerups(self, flattened_level: str) -> int:
        return np.sum([flattened_level.count(char) for char in ["1", "?", "U", "L"]])

    def count_coins(self, flattened_level: str) -> int:
        return flattened_level.count("o")

    def _flatten_level(self, string_level: List[str]) -> str:
        return "".join(string_level)

    def pipe_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_pipes(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.pipe_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} pipes", keyword

    def enemy_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_enemies(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.enemy_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} enemies", keyword

    def block_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_blocks(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.block_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} blocks", keyword

    def koopa_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_koopas(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.block_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} koopas", keyword

    def goomba_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_blocks(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.block_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} goombas", keyword

    def powerup_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_powerups(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.block_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} powerups", keyword

    def coin_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_coins(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.block_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} coins", keyword

    def elevation_prompt(self, flattened_level: str, level: str):
        top_levels = level[:6]  # elevation 8 and up
        for t in top_levels:
            if "X" in t or "<" in t or ">" in t:
                return "high elevation", "high"
        return "low elevation", "low"

    def output_hidden(self, prompt: str, device: torch.device = torch.device("cpu")):
        # Reducing along the first dimension to get a 768 dimensional array
        return (
            self.feature_extraction(prompt, return_tensors="pt")[0]
            .mean(0)
            .to(device)
            .view(1, -1)
        )

    def dataset_statistics(self, dataset: MarioDataset):
        enemy_counts = []
        pipe_counts = []
        block_counts = []
        goomba_counts = []
        koopa_counts = []
        coin_counts = []
        powerup_counts = []
        for i in range(len(dataset)):
            level, _ = dataset[i]
            str_level = self._flatten_level(view_level(level, dataset.tokenizer))

            enemy_count = self.count_enemies(str_level)
            pipe_count = self.count_pipes(str_level)
            block_count = self.count_blocks(str_level)
            goomba_count = self.count_goombas(str_level)
            koopa_count = self.count_koopas(str_level)
            coin_count = self.count_coins(str_level)
            powerup_count = self.count_powerups(str_level)

            enemy_counts.append(enemy_count)
            pipe_counts.append(pipe_count)
            block_counts.append(block_count)
            goomba_counts.append(goomba_count)
            koopa_counts.append(koopa_count)
            coin_counts.append(coin_count)
            powerup_counts.append(powerup_count)
        d = {"enemy": {}, "pipe": {}, "block": {}, "goomba": {}, "koopa": {}, "coin": {}, "powerup": {}}

        d["enemy"] = stats.mstats.mquantiles(enemy_counts, [0.33, 0.66, 0.95])
        d["pipe"] = stats.mstats.mquantiles(pipe_counts, [0.33, 0.66, 0.95])
        d["block"] = stats.mstats.mquantiles(block_counts, [0.33, 0.66, 0.95])
        d["goomba"] = stats.mstats.mquantiles(goomba_counts, [0.33, 0.66, 0.95])
        d["koopa"] = stats.mstats.mquantiles(koopa_counts, [0.33, 0.66, 0.95])
        d["coin"] = stats.mstats.mquantiles(coin_counts, [0.33, 0.66, 0.95])
        d["powerup"] = stats.mstats.mquantiles(powerup_counts, [0.33, 0.66, 0.95])
        return d

    def __call__(
        self, level: torch.Tensor = None, sample_prompt: bool = False
    ) -> Union[str, torch.Tensor]:
        device: torch.device = torch.device("cpu")
        if not sample_prompt:
            if level is None:
                raise ValueError("Level must be provided if sample_prompt is not true!")
            str_level = view_level(level, self.level_tokenizer)
            flattened_level = self._flatten_level(str_level)

            pipe_prompt, _ = self.pipe_prompt(flattened_level, str_level)
            enemy_prompt, _ = self.enemy_prompt(flattened_level, str_level)
            block_prompt, _ = self.block_prompt(flattened_level, str_level)
            goomba_prompt, _ = self.goomba_prompt(flattened_level, str_level)
            koopa_prompt, _ = self.koopa_prompt(flattened_level, str_level)
            coin_prompt, _ = self.coin_prompt(flattened_level, str_level)
            powerup_prompt, _ = self.powerup_prompt(flattened_level, str_level)
            elevation_prompt, _ = self.elevation_prompt(flattened_level, str_level)
            device = level.device
        else:
            str_level = None
            pipe_prompt = random.choice(["no", "little", "some", "many"]) + " pipes"
            enemy_prompt = random.choice(["no", "little", "some", "many"]) + " enemies"
            block_prompt = (
                random.choice(["little", "little", "some", "many"]) + " blocks"
            )  # levels always have blocks
            goomba_prompt = random.choice(["no", "little", "some", "many"]) + " goombas"
            koopa_prompt = random.choice(["no", "little", "some", "many"]) + " koopas"
            coin_prompt = random.choice(["no", "little", "some", "many"]) + " coins"
            powerup_prompt = random.choice(["no", "little", "some", "many"]) + " powerups"
            elevation_prompt = (
                random.choice(["low", "high"]) + " elevation"
            )  # levels always have blocks

        prompt_dict = {
            "pipe": pipe_prompt,
            "enemy": enemy_prompt,
            "block": block_prompt,
            "goomba": goomba_prompt,
            "koopa": koopa_prompt,
            "coin": coin_prompt,
            "powerup":powerup_prompt,
            "elevation_prompt": elevation_prompt,
        }
        prompt = f"{pipe_prompt}, {enemy_prompt}, {block_prompt}, {goomba_prompt}, {koopa_prompt}, {coin_prompt}, {powerup_prompt}, {elevation_prompt}"
        hidden = self.output_hidden(prompt, device=device)
        return prompt, hidden, prompt_dict, str_level
