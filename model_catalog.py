from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional


class HardwarePlatform(Enum):
    MACOS = "macos"
    CUDA = "cuda"
    CPU = "cpu"

    @classmethod
    def current(cls) -> "HardwarePlatform":
        if sys.platform == "darwin":
            return cls.MACOS
        if cls._is_cuda_available():
            return cls.CUDA
        return cls.CPU

    @staticmethod
    def _is_cuda_available() -> bool:
        try:
            import torch
        except ImportError:
            return False
        return torch.cuda.is_available()


@dataclass(frozen=True)
class ChatModelOption:
    repository_id: str
    display_name: str
    parameter_size: str
    description: str
    requires_token: bool = False
    mlx_compatible: bool = True


DEFAULT_MODEL_OPTIONS: List[ChatModelOption] = [
    ChatModelOption(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen2.5 0.5B Instruct",
        "0.5B",
        "Fastest option, best for a first smoke test.",
        mlx_compatible=False,
    ),
    ChatModelOption(
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2.5 1.5B Instruct",
        "1.5B",
        "Small and capable chat model.",
        mlx_compatible=False,
    ),
    ChatModelOption(
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen2.5 3B Instruct",
        "3B",
        "Better quality, still modest disk usage.",
        mlx_compatible=False,
    ),
    ChatModelOption(
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5 7B Instruct",
        "7B",
        "Stronger model, more disk and slower per token.",
        mlx_compatible=False,
    ),
    ChatModelOption(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "TinyLlama 1.1B Chat",
        "1.1B",
        "Compact Llama-style chat model.",
    ),
    ChatModelOption(
        "meta-llama/Llama-3.2-3B-Instruct",
        "Llama 3.2 3B Instruct",
        "3B",
        "Gated on Hugging Face, requires an access token.",
        requires_token=True,
    ),
    ChatModelOption(
        "meta-llama/Llama-3.1-8B-Instruct",
        "Llama 3.1 8B Instruct",
        "8B",
        "Gated on Hugging Face, requires an access token.",
        requires_token=True,
    ),
    ChatModelOption(
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Mistral 7B Instruct v0.3",
        "7B",
        "Gated on Hugging Face, requires an access token.",
        requires_token=True,
    ),
]


class ModelCatalog:
    def __init__(self, options: List[ChatModelOption]) -> None:
        self._options = options

    @classmethod
    def all_models(cls) -> "ModelCatalog":
        return cls(list(DEFAULT_MODEL_OPTIONS))

    @classmethod
    def mlx_compatible_models(cls) -> "ModelCatalog":
        return cls(
            [option for option in DEFAULT_MODEL_OPTIONS if option.mlx_compatible]
        )

    @property
    def options(self) -> List[ChatModelOption]:
        return list(self._options)

    @property
    def default_option(self) -> ChatModelOption:
        return self._options[0]

    def find_by_repository_id(self, repository_id: str) -> Optional[ChatModelOption]:
        normalized_repository_id = repository_id.strip().lower()
        for option in self._options:
            if option.repository_id.lower() == normalized_repository_id:
                return option
        return None


class InteractiveModelSelector:
    def __init__(
        self,
        catalog: ModelCatalog,
        input_reader: Callable[[str], str] = input,
        output_stream: object = sys.stdout,
    ) -> None:
        self._catalog = catalog
        self._input_reader = input_reader
        self._output_stream = output_stream

    def select(self) -> ChatModelOption:
        options = self._catalog.options
        self._print_options(options)
        while True:
            raw_value = self._input_reader("Select a model [1]: ").strip()
            if not raw_value:
                return options[0]
            if raw_value.isdigit():
                index = int(raw_value)
                if 1 <= index <= len(options):
                    return options[index - 1]
                self._write(f"Enter a number between 1 and {len(options)}.")
                continue
            known_option = self._catalog.find_by_repository_id(raw_value)
            if known_option is not None:
                return known_option
            return ChatModelOption(
                repository_id=raw_value,
                display_name=raw_value,
                parameter_size="custom",
                description="Custom Hugging Face model.",
            )

    def _print_options(self, options: List[ChatModelOption]) -> None:
        self._write("Available models:")
        for position, option in enumerate(options, start=1):
            token_note = " (gated)" if option.requires_token else ""
            self._write(
                f"  {position}. {option.display_name} - {option.parameter_size}"
                f"{token_note}: {option.description}"
            )
        self._write("Enter a number or a Hugging Face repository id.")

    def _write(self, message: str) -> None:
        print(message, file=self._output_stream, flush=True)
