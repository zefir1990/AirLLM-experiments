#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.util
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple

from model_catalog import (
    ChatModelOption,
    HardwarePlatform,
    InteractiveModelSelector,
    ModelCatalog,
)

DEFAULT_MAX_LENGTH = 2048
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_CONTEXT_CHARACTER_BUDGET = 8000

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}

STOP_SEQUENCES = (
    "<|im_end|>",
    "<|im_start|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end|>",
    "<|endoftext|>",
    "</s>",
)

HUGGING_FACE_TOKEN_ENVIRONMENT_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


class StopSequenceFilter:
    def __init__(self, stop_sequences: Tuple[str, ...]) -> None:
        self._stop_sequences = stop_sequences
        self._pending_text = ""
        self._has_stopped = False

    @property
    def has_stopped(self) -> bool:
        return self._has_stopped

    def feed(self, text: str) -> str:
        if self._has_stopped:
            return ""
        self._pending_text += text
        stop_position = self._find_stop_position()
        if stop_position is not None:
            return self._stop_at(stop_position)
        return self._release_safe_prefix()

    def flush(self) -> str:
        if self._has_stopped:
            return ""
        return self._take(len(self._pending_text))

    def _find_stop_position(self) -> Optional[int]:
        found_positions = [
            self._pending_text.find(stop_sequence)
            for stop_sequence in self._stop_sequences
        ]
        found_positions = [
            position for position in found_positions if position != -1
        ]
        return min(found_positions) if found_positions else None

    def _stop_at(self, stop_position: int) -> str:
        self._has_stopped = True
        return self._take(stop_position)

    def _release_safe_prefix(self) -> str:
        return self._take(self._safe_length())

    def _safe_length(self) -> int:
        safe_length = len(self._pending_text)
        for stop_sequence in self._stop_sequences:
            for prefix_length in range(1, len(stop_sequence) + 1):
                if self._pending_text.endswith(stop_sequence[:prefix_length]):
                    safe_length = min(
                        safe_length, len(self._pending_text) - prefix_length
                    )
        return safe_length

    def _take(self, length: int) -> str:
        taken_text = self._pending_text[:length]
        self._pending_text = self._pending_text[length:]
        return taken_text


class ApplicationError(Exception):
    pass


class InstallationError(ApplicationError):
    pass


class ModelLoadingError(ApplicationError):
    pass


class GenerationError(ApplicationError):
    pass


@dataclass(frozen=True)
class Dependency:
    import_name: str
    pip_requirement: str


class DependencyInstaller:
    def __init__(self, output_stream: object = sys.stdout) -> None:
        self._output_stream = output_stream

    def ensure_dependencies(self) -> None:
        for dependency in self._required_dependencies():
            if self._is_importable(dependency.import_name):
                continue
            self._install(dependency)

    def _required_dependencies(self) -> List[Dependency]:
        dependencies = [Dependency("airllm", "airllm")]
        if HardwarePlatform.current() is HardwarePlatform.MACOS:
            dependencies.append(Dependency("mlx", "mlx"))
        return dependencies

    @staticmethod
    def _is_importable(import_name: str) -> bool:
        return importlib.util.find_spec(import_name) is not None

    def _install(self, dependency: Dependency) -> None:
        if self._run_install(dependency, use_user_site=False) != 0:
            if self._run_install(dependency, use_user_site=True) != 0:
                raise InstallationError(
                    f"Failed to install '{dependency.pip_requirement}'."
                )
        importlib.invalidate_caches()

    def _run_install(self, dependency: Dependency, use_user_site: bool) -> int:
        arguments = [sys.executable, "-m", "pip", "install", "--upgrade"]
        if use_user_site:
            arguments.append("--user")
        arguments.append(dependency.pip_requirement)
        self._write(f"$ {' '.join(shlex.quote(argument) for argument in arguments)}")
        return subprocess.run(arguments).returncode

    def _write(self, message: str) -> None:
        print(message, file=self._output_stream, flush=True)


@dataclass(frozen=True)
class GenerationSettings:
    max_length: int = DEFAULT_MAX_LENGTH
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE


class ChatEngine:
    def __init__(
        self,
        option: ChatModelOption,
        settings: GenerationSettings,
        output_stream: object = sys.stdout,
    ) -> None:
        self._option = option
        self._settings = settings
        self._output_stream = output_stream
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        self._write(f"Loading {self._option.display_name} ...")
        self._write(
            "The first load downloads the model and splits it into layer shards; "
            "this can take a long time."
        )
        try:
            from airllm import AutoModel
        except ImportError as error:
            raise ModelLoadingError(f"AirLLM is not importable: {error}") from error
        try:
            self._model = AutoModel.from_pretrained(
                self._option.repository_id, **self._model_kwargs()
            )
        except Exception as error:
            raise ModelLoadingError(
                f"Failed to load '{self._option.repository_id}': {error}"
            ) from error
        self._tokenizer = self._model.tokenizer

    def stream(self, messages: List[dict]) -> Iterator[str]:
        if self._model is None or self._tokenizer is None:
            raise GenerationError("The model is not loaded.")
        prompt = self._compose_prompt(messages)
        try:
            yield from self._filtered_stream(prompt)
        except GenerationError:
            raise
        except Exception as error:
            raise GenerationError(str(error)) from error

    def _filtered_stream(self, prompt: str) -> Iterator[str]:
        stop_sequence_filter = StopSequenceFilter(STOP_SEQUENCES)
        for piece in self._stream_text(prompt):
            filtered_piece = stop_sequence_filter.feed(piece)
            if filtered_piece:
                yield filtered_piece
            if stop_sequence_filter.has_stopped:
                return
        remaining_piece = stop_sequence_filter.flush()
        if remaining_piece:
            yield remaining_piece

    def _model_kwargs(self) -> dict:
        model_kwargs = {"max_seq_len": self._settings.max_length}
        token = self._hugging_face_token()
        if token:
            model_kwargs["hf_token"] = token
        return model_kwargs

    def _compose_prompt(self, messages: List[dict]) -> str:
        if self._supports_chat_template():
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return self._compose_plain_prompt(messages)

    def _supports_chat_template(self) -> bool:
        return bool(getattr(self._tokenizer, "chat_template", None))

    @staticmethod
    def _compose_plain_prompt(messages: List[dict]) -> str:
        lines = [
            f"{message['role'].capitalize()}: {message['content']}"
            for message in messages
        ]
        lines.append("Assistant:")
        return "\n".join(lines)

    @staticmethod
    def _hugging_face_token() -> Optional[str]:
        for environment_variable in HUGGING_FACE_TOKEN_ENVIRONMENT_VARIABLES:
            token = os.environ.get(environment_variable)
            if token:
                return token
        return None

    def _stream_text(self, prompt: str) -> Iterator[str]:
        raise NotImplementedError

    def _write(self, message: str) -> None:
        print(message, file=self._output_stream, flush=True)


class MlxChatEngine(ChatEngine):
    def _stream_text(self, prompt: str) -> Iterator[str]:
        import mlx.core as mx

        tokenized = self._tokenizer(
            [prompt],
            return_tensors="np",
            return_attention_mask=False,
            truncation=True,
            max_length=self._settings.max_length,
        )
        input_array = mx.array(tokenized["input_ids"])
        generated_token_ids: List[int] = []
        emitted_text = ""
        for token in self._model.model_generate(
            input_array, temperature=self._settings.temperature
        ):
            generated_token_ids.append(int(token.item()))
            full_text = self._tokenizer.decode(generated_token_ids)
            delta_text = full_text[len(emitted_text) :]
            emitted_text = full_text
            if delta_text:
                yield delta_text
            if len(generated_token_ids) >= self._settings.max_new_tokens:
                break


class TorchChatEngine(ChatEngine):
    def _model_kwargs(self) -> dict:
        model_kwargs = super()._model_kwargs()
        model_kwargs["device"] = self._device
        return model_kwargs

    def _stream_text(self, prompt: str) -> Iterator[str]:
        import torch

        tokenized = self._tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=self._settings.max_length,
        )
        input_ids = tokenized["input_ids"].to(self._device)
        generation_kwargs = {
            "max_new_tokens": self._settings.max_new_tokens,
            "use_cache": True,
        }
        if self._settings.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self._settings.temperature
        else:
            generation_kwargs["do_sample"] = False
        with torch.no_grad():
            generated = self._model.generate(input_ids, **generation_kwargs)
        generated_token_ids = generated[0][input_ids.shape[1] :].tolist()
        emitted_text = ""
        emitted_token_ids: List[int] = []
        for token_id in generated_token_ids:
            emitted_token_ids.append(token_id)
            full_text = self._tokenizer.decode(
                emitted_token_ids, skip_special_tokens=True
            )
            delta_text = full_text[len(emitted_text) :]
            emitted_text = full_text
            if delta_text:
                yield delta_text

    @property
    def _device(self) -> str:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"


class ChatEngineFactory:
    def __init__(
        self,
        settings: GenerationSettings,
        output_stream: object = sys.stdout,
    ) -> None:
        self._settings = settings
        self._output_stream = output_stream

    def create(self, option: ChatModelOption) -> ChatEngine:
        if HardwarePlatform.current() is HardwarePlatform.MACOS:
            return MlxChatEngine(option, self._settings, self._output_stream)
        return TorchChatEngine(option, self._settings, self._output_stream)


class InteractiveChatSession:
    def __init__(
        self,
        chat_engine: ChatEngine,
        option: ChatModelOption,
        input_reader: Callable[[str], str] = input,
        output_stream: object = sys.stdout,
    ) -> None:
        self._chat_engine = chat_engine
        self._option = option
        self._input_reader = input_reader
        self._output_stream = output_stream
        self._messages: List[dict] = []

    def run(self) -> None:
        self._print_welcome()
        while True:
            try:
                raw_input = self._input_reader("\nYou: ")
            except (EOFError, KeyboardInterrupt):
                self._write("\nGoodbye.")
                return

            message = raw_input.strip()
            if not message:
                continue
            if message.startswith("/") or message.lower() in EXIT_COMMANDS:
                if not self._handle_command(message):
                    return
                continue
            self._respond(message)

    def _respond(self, message: str) -> None:
        self._messages.append({"role": "user", "content": message})
        self._write("\nAssistant: ")
        collected_content: List[str] = []
        try:
            for content in self._chat_engine.stream(self._limited_messages()):
                collected_content.append(content)
                self._write("".join(collected_content))
        except GenerationError as error:
            self._messages.pop()
            self._write(f"\n[error] {error}")
            return
        except KeyboardInterrupt:
            self._messages.pop()
            self._write("\n[interrupted]")
            return
        self._messages.append(
            {"role": "assistant", "content": "".join(collected_content)}
        )

    def _limited_messages(self) -> List[dict]:
        limited_messages = list(self._messages)
        while (
            len(limited_messages) > 1
            and self._character_count(limited_messages) > DEFAULT_CONTEXT_CHARACTER_BUDGET
        ):
            removal_index = 1 if limited_messages[0]["role"] == "system" else 0
            limited_messages.pop(removal_index)
        return limited_messages

    @staticmethod
    def _character_count(messages: List[dict]) -> int:
        return sum(len(message["content"]) for message in messages)

    def _handle_command(self, command: str) -> bool:
        normalized_command = command.lower()
        if normalized_command in EXIT_COMMANDS:
            self._write("Goodbye.")
            return False
        if normalized_command == "/clear":
            self._messages.clear()
            self._write("Conversation cleared.")
            return True
        if normalized_command == "/help":
            self._print_help()
            return True
        self._write(f"Unknown command: {command}. Type /help for available commands.")
        return True

    def _print_welcome(self) -> None:
        token_note = (
            " This model is gated; set HF_TOKEN before starting."
            if self._option.requires_token
            else ""
        )
        self._write(
            f"Chatting with {self._option.display_name} ({self._option.repository_id})."
            f"{token_note}\n"
            "The first response can take minutes while the model is downloaded and split.\n"
            "Type /help for commands, or press Ctrl+C to interrupt a response."
        )

    def _print_help(self) -> None:
        self._write(
            "Commands:\n"
            "  /help   Show this help\n"
            "  /clear  Clear the conversation history\n"
            "  /exit   Leave the chat\n"
            "  Ctrl+C  Interrupt a response"
        )

    def _write(self, text: str) -> None:
        print(text, file=self._output_stream, flush=True)


class Application:
    def __init__(
        self,
        input_reader: Callable[[str], str] = input,
        output_stream: object = sys.stdout,
    ) -> None:
        self._input_reader = input_reader
        self._output_stream = output_stream

    def _catalog_for(self, platform: HardwarePlatform) -> ModelCatalog:
        if platform is HardwarePlatform.MACOS:
            print(
                "On macOS AirLLM runs through its MLX Llama runtime, so only "
                "Llama-style models (no attention bias, untied embeddings) are listed.",
                file=self._output_stream,
                flush=True,
            )
            return ModelCatalog.mlx_compatible_models()
        return ModelCatalog.all_models()

    def run(self) -> None:
        platform = HardwarePlatform.current()
        option = InteractiveModelSelector(
            self._catalog_for(platform),
            self._input_reader,
            self._output_stream,
        ).select()

        DependencyInstaller(self._output_stream).ensure_dependencies()

        chat_engine = ChatEngineFactory(
            GenerationSettings(), self._output_stream
        ).create(option)
        chat_engine.load()

        InteractiveChatSession(
            chat_engine,
            option,
            self._input_reader,
            self._output_stream,
        ).run()


def main() -> int:
    application = Application()
    try:
        application.run()
    except ApplicationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
