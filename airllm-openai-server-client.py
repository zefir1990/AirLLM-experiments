#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterator, List, Optional

import requests


DEFAULT_REPOSITORY_URL = "https://github.com/mkamranr/airllm-openai-server.git"
DEFAULT_SOURCE_DIRECTORY = "airllm-openai-server"
DEFAULT_DOCKERFILE_PATH = "docker/Dockerfile"
DEFAULT_IMAGE_TAG = "airllm-server:cpu"
DEFAULT_CONTAINER_NAME = "airllm-server"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_HOST_PORT = 8000
DEFAULT_CONTAINER_PORT = 8000
DEFAULT_VOLUME_NAME = "airllm-cache"
DEFAULT_API_KEY = "not-needed"
DEFAULT_HEALTH_TIMEOUT_SECONDS = 180.0
DEFAULT_HEALTH_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 900.0

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}


class ApplicationError(Exception):
    pass


class CommandExecutionError(ApplicationError):
    pass


class InstallationError(ApplicationError):
    pass


class ChatClientError(ApplicationError):
    pass


class CommandExecutor:
    def __init__(self, output_stream: object = sys.stdout) -> None:
        self._output_stream = output_stream

    def run(
        self,
        arguments: List[str],
        working_directory: Optional[str] = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess:
        self._echo(arguments)
        return subprocess.run(
            arguments,
            cwd=working_directory,
            capture_output=capture_output,
            text=True,
        )

    def run_checked(
        self,
        arguments: List[str],
        working_directory: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        completed_process = self.run(arguments, working_directory=working_directory)
        if completed_process.returncode != 0:
            pretty_command = " ".join(arguments)
            raise CommandExecutionError(
                f"Command failed with exit code {completed_process.returncode}: {pretty_command}"
            )
        return completed_process

    def _echo(self, arguments: List[str]) -> None:
        pretty_command = " ".join(shlex.quote(argument) for argument in arguments)
        print(f"$ {pretty_command}", file=self._output_stream, flush=True)


@dataclass(frozen=True)
class ServerConfiguration:
    repository_url: str
    source_directory: str
    dockerfile_path: str
    image_tag: str
    container_name: str
    model_name: str
    host_port: int
    container_port: int
    volume_name: str
    huggingface_token: Optional[str] = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.host_port}"

    @classmethod
    def with_defaults(cls) -> "ServerConfiguration":
        return cls(
            repository_url=DEFAULT_REPOSITORY_URL,
            source_directory=DEFAULT_SOURCE_DIRECTORY,
            dockerfile_path=DEFAULT_DOCKERFILE_PATH,
            image_tag=DEFAULT_IMAGE_TAG,
            container_name=DEFAULT_CONTAINER_NAME,
            model_name=DEFAULT_MODEL_NAME,
            host_port=DEFAULT_HOST_PORT,
            container_port=DEFAULT_CONTAINER_PORT,
            volume_name=DEFAULT_VOLUME_NAME,
        )


class InteractiveConfigurator:
    def __init__(
        self,
        input_reader: Callable[[str], str] = input,
        output_stream: object = sys.stdout,
    ) -> None:
        self._input_reader = input_reader
        self._output_stream = output_stream

    def configure(self) -> ServerConfiguration:
        self._write(
            "\nAirLLM OpenAI server is not installed yet.\n"
            "Press Enter to accept each default value.\n"
        )
        repository_url = self._prompt("Repository URL", DEFAULT_REPOSITORY_URL)
        source_directory = self._prompt("Source directory", DEFAULT_SOURCE_DIRECTORY)
        dockerfile_path = self._prompt("Dockerfile path", DEFAULT_DOCKERFILE_PATH)
        image_tag = self._prompt("Docker image tag", DEFAULT_IMAGE_TAG)
        container_name = self._prompt("Container name", DEFAULT_CONTAINER_NAME)
        model_name = self._prompt("Model name", DEFAULT_MODEL_NAME)
        host_port = self._prompt_integer("Host port", DEFAULT_HOST_PORT)
        container_port = self._prompt_integer("Container port", DEFAULT_CONTAINER_PORT)
        volume_name = self._prompt("Cache volume name", DEFAULT_VOLUME_NAME)
        huggingface_token = self._prompt_optional("HuggingFace token (optional)")

        return ServerConfiguration(
            repository_url=repository_url,
            source_directory=source_directory,
            dockerfile_path=dockerfile_path,
            image_tag=image_tag,
            container_name=container_name,
            model_name=model_name,
            host_port=host_port,
            container_port=container_port,
            volume_name=volume_name,
            huggingface_token=huggingface_token,
        )

    def _prompt(self, label: str, default: str) -> str:
        raw_value = self._input_reader(f"{label} [{default}]: ").strip()
        return raw_value or default

    def _prompt_optional(self, label: str) -> Optional[str]:
        raw_value = self._input_reader(f"{label}: ").strip()
        return raw_value or None

    def _prompt_integer(self, label: str, default: int) -> int:
        while True:
            raw_value = self._prompt(label, str(default))
            try:
                return int(raw_value)
            except ValueError:
                self._write("Please enter a valid integer.")

    def _write(self, message: str) -> None:
        print(message, file=self._output_stream, flush=True)


class DockerCli:
    def __init__(self, command_executor: CommandExecutor) -> None:
        self._command_executor = command_executor

    def ensure_available(self) -> None:
        if shutil.which("docker") is None:
            raise InstallationError(
                "Docker is not installed or not on PATH. Install Docker and retry."
            )
        completed_process = self._command_executor.run(
            ["docker", "info"], capture_output=True
        )
        if completed_process.returncode != 0:
            raise InstallationError(
                "Docker is installed but the daemon is not reachable. Start Docker and retry."
            )

    def image_exists(self, image_tag: str) -> bool:
        return self._inspect(["docker", "image", "inspect", image_tag])

    def container_exists(self, container_name: str) -> bool:
        return self._inspect(["docker", "container", "inspect", container_name])

    def container_is_running(self, container_name: str) -> bool:
        completed_process = self._command_executor.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
            capture_output=True,
        )
        if completed_process.returncode != 0:
            return False
        return completed_process.stdout.strip() == "true"

    def find_container_by_image(self, image_tag: str) -> Optional[str]:
        running_container_names = self._list_container_names(
            ["docker", "ps", "--filter", f"ancestor={image_tag}", "--format", "{{.Names}}"]
        )
        if running_container_names:
            return running_container_names[0]
        stopped_container_names = self._list_container_names(
            ["docker", "ps", "--all", "--filter", f"ancestor={image_tag}", "--format", "{{.Names}}"]
        )
        return stopped_container_names[0] if stopped_container_names else None

    def _list_container_names(self, arguments: List[str]) -> List[str]:
        completed_process = self._command_executor.run(arguments, capture_output=True)
        if completed_process.returncode != 0:
            return []
        return completed_process.stdout.split()

    def build_image(self, configuration: ServerConfiguration) -> None:
        self._command_executor.run_checked(
            [
                "docker",
                "build",
                "-f",
                configuration.dockerfile_path,
                "-t",
                configuration.image_tag,
                ".",
            ],
            working_directory=configuration.source_directory,
        )

    def run_container(self, configuration: ServerConfiguration) -> None:
        arguments = [
            "docker",
            "run",
            "-d",
            "--name",
            configuration.container_name,
            "-p",
            f"{configuration.host_port}:{configuration.container_port}",
            "-v",
            f"{configuration.volume_name}:/cache",
            "-e",
            f"AIRLLM_MODEL={configuration.model_name}",
        ]
        if configuration.huggingface_token:
            arguments.extend(["-e", f"HF_TOKEN={configuration.huggingface_token}"])
        arguments.append(configuration.image_tag)
        self._command_executor.run_checked(arguments)

    def start_container(self, container_name: str) -> None:
        self._command_executor.run_checked(["docker", "start", container_name])

    def _inspect(self, arguments: List[str]) -> bool:
        completed_process = self._command_executor.run(arguments, capture_output=True)
        return completed_process.returncode == 0


class SourceRepository:
    def __init__(self, command_executor: CommandExecutor) -> None:
        self._command_executor = command_executor

    def ensure_available(self, configuration: ServerConfiguration) -> None:
        if os.path.isdir(configuration.source_directory):
            return
        if shutil.which("git") is None:
            raise InstallationError(
                "git is not installed or not on PATH and the source directory is missing."
            )
        self._command_executor.run_checked(
            [
                "git",
                "clone",
                "--depth",
                "1",
                configuration.repository_url,
                configuration.source_directory,
            ]
        )


class ServerProvisioner:
    def __init__(
        self,
        docker_cli: DockerCli,
        source_repository: SourceRepository,
        configuration: ServerConfiguration,
        output_stream: object = sys.stdout,
    ) -> None:
        self._docker_cli = docker_cli
        self._source_repository = source_repository
        self._configuration = configuration
        self._output_stream = output_stream

    def ensure_installed_and_running(self) -> None:
        self._docker_cli.ensure_available()

        container_name = self._configuration.container_name
        if self._docker_cli.container_exists(container_name):
            self._write(f"Container '{container_name}' already exists.")
            if self._docker_cli.container_is_running(container_name):
                self._write(f"Container '{container_name}' is already running.")
                return
            self._write(f"Starting container '{container_name}'.")
            self._docker_cli.start_container(container_name)
            return

        if self._docker_cli.image_exists(self._configuration.image_tag):
            self._write(f"Image '{self._configuration.image_tag}' already exists.")
        else:
            self._write(f"Image '{self._configuration.image_tag}' not found. Building it.")
            self._source_repository.ensure_available(self._configuration)
            self._docker_cli.build_image(self._configuration)

        self._write(f"Creating container '{container_name}'.")
        self._docker_cli.run_container(self._configuration)

    def _write(self, message: str) -> None:
        print(message, file=self._output_stream, flush=True)


class ServerHealthWaiter:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_HEALTH_POLL_INTERVAL_SECONDS,
        output_stream: object = sys.stdout,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._output_stream = output_stream

    def wait_until_ready(self) -> None:
        health_url = f"{self._base_url}/health"
        self._write(f"Waiting for server health at {health_url} ...")
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            if self._is_healthy(health_url):
                self._write("Server is ready.")
                return
            time.sleep(self._poll_interval_seconds)
        raise ApplicationError(
            f"Server did not become ready within {self._timeout_seconds:.0f} seconds."
        )

    def _is_healthy(self, health_url: str) -> bool:
        try:
            response = requests.get(health_url, timeout=5)
        except requests.RequestException:
            return False
        return response.status_code == 200

    def _write(self, message: str) -> None:
        print(message, file=self._output_stream, flush=True)


class OpenAiCompatibleChatClient:
    def __init__(
        self,
        base_url: str,
        configured_model_name: Optional[str] = None,
        api_key: str = DEFAULT_API_KEY,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._configured_model_name = configured_model_name
        self._api_key = api_key
        self._request_timeout_seconds = request_timeout_seconds
        self._resolved_model_name: Optional[str] = None

    def resolve_model_name(self) -> str:
        if self._resolved_model_name is not None:
            return self._resolved_model_name
        discovered_model_name = self._discover_model_name()
        self._resolved_model_name = discovered_model_name or self._configured_model_name
        if not self._resolved_model_name:
            raise ChatClientError("No model is available from the server.")
        return self._resolved_model_name

    def stream_completion(self, messages: List[dict]) -> Iterator[str]:
        model_name = self.resolve_model_name()
        request_payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
        }
        try:
            response = requests.post(
                f"{self._base_url}/v1/chat/completions",
                headers=self._headers(),
                json=request_payload,
                stream=True,
                timeout=self._request_timeout_seconds,
            )
        except requests.RequestException as error:
            raise ChatClientError(f"Request failed: {error}") from error

        with response:
            if response.status_code != 200:
                raise ChatClientError(self._format_error(response))
            for content in self._iter_stream_content(response):
                yield content

    def _discover_model_name(self) -> Optional[str]:
        try:
            response = requests.get(
                f"{self._base_url}/v1/models",
                headers=self._headers(),
                timeout=10,
            )
            response.raise_for_status()
            models = response.json().get("data", [])
        except (requests.RequestException, ValueError):
            return None
        if not models:
            return None
        return models[0].get("id")

    def _iter_stream_content(self, response: requests.Response) -> Iterator[str]:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    yield content

    def _format_error(self, response: requests.Response) -> str:
        try:
            payload = response.json()
            error_message = payload.get("error", {}).get("message")
        except ValueError:
            error_message = None
        if error_message:
            return f"Server error ({response.status_code}): {error_message}"
        return f"Server error ({response.status_code}): {response.text.strip()}"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }


class InteractiveChatSession:
    def __init__(
        self,
        chat_client: OpenAiCompatibleChatClient,
        model_name: str,
        input_reader: Callable[[str], str] = input,
        output_stream: object = sys.stdout,
    ) -> None:
        self._chat_client = chat_client
        self._model_name = model_name
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

    def _respond(self, message: str) -> None:
        self._messages.append({"role": "user", "content": message})
        self._write("\nAssistant: ")
        collected_content: List[str] = []
        try:
            for content in self._chat_client.stream_completion(self._messages):
                collected_content.append(content)
                self._write(content, end="")
        except ChatClientError as error:
            self._messages.pop()
            self._write(f"\n[error] {error}")
            return
        self._write("")
        self._messages.append(
            {"role": "assistant", "content": "".join(collected_content)}
        )

    def _print_welcome(self) -> None:
        self._write(
            f"Connected to model: {self._model_name}\n"
            "The first reply may take minutes while the model is downloaded and split.\n"
            "Type /help for commands."
        )

    def _print_help(self) -> None:
        self._write(
            "Commands:\n"
            "  /help   Show this help\n"
            "  /clear  Clear the conversation history\n"
            "  /exit   Leave the chat"
        )

    def _write(self, text: str, end: str = "\n") -> None:
        print(text, end=end, file=self._output_stream, flush=True)


class Application:
    def __init__(self, output_stream: object = sys.stdout) -> None:
        self._output_stream = output_stream
        self._command_executor = CommandExecutor(output_stream)
        self._docker_cli = DockerCli(self._command_executor)

    def run(self) -> None:
        self._docker_cli.ensure_available()
        configuration = self._resolve_configuration()
        provisioner = ServerProvisioner(
            self._docker_cli,
            SourceRepository(self._command_executor),
            configuration,
            self._output_stream,
        )
        provisioner.ensure_installed_and_running()

        ServerHealthWaiter(configuration.base_url, output_stream=self._output_stream).wait_until_ready()

        chat_client = OpenAiCompatibleChatClient(
            configuration.base_url,
            configured_model_name=configuration.model_name,
        )
        model_name = chat_client.resolve_model_name()
        InteractiveChatSession(
            chat_client,
            model_name,
            output_stream=self._output_stream,
        ).run()

    def _resolve_configuration(self) -> ServerConfiguration:
        default_configuration = ServerConfiguration.with_defaults()
        existing_container_name = self._find_existing_container_name(default_configuration)
        if existing_container_name is not None:
            print(
                f"Existing container '{existing_container_name}' detected. "
                "Using default configuration.",
                file=self._output_stream,
                flush=True,
            )
            return replace(default_configuration, container_name=existing_container_name)
        return InteractiveConfigurator(output_stream=self._output_stream).configure()

    def _find_existing_container_name(
        self, default_configuration: ServerConfiguration
    ) -> Optional[str]:
        if self._docker_cli.container_exists(DEFAULT_CONTAINER_NAME):
            return DEFAULT_CONTAINER_NAME
        return self._docker_cli.find_container_by_image(default_configuration.image_tag)


def main() -> int:
    application = Application()
    try:
        application.run()
    except ApplicationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
