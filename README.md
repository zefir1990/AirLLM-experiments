# AirLLM experiments

Interactive CLI that installs and chats with an
[airllm-openai-server](https://github.com/mkamranr/airllm-openai-server) running in Docker.

On first run it configures and launches the server, then opens a streaming chat session
against its OpenAI-compatible API. On later runs it reuses the existing container.

## Requirements

- Python 3.7+
- [Docker](https://docs.docker.com/get-docker/) running locally
- `git` (only needed when the server source must be cloned)
- Python package `requests`

```sh
python3 -m pip install requests
```

## Usage

```sh
python3 main.py
```

### First run

The server is not installed, so the tool prompts for the installation settings. Press
Enter to accept each default:

| Setting          | Default                                              |
| ---------------- | ---------------------------------------------------- |
| Repository URL   | `https://github.com/mkamranr/airllm-openai-server.git` |
| Source directory | `airllm-openai-server`                               |
| Dockerfile path  | `docker/Dockerfile`                                  |
| Image tag        | `airllm-server:cpu`                                  |
| Container name   | `airllm-server`                                      |
| Model name       | `Qwen/Qwen2.5-0.5B-Instruct`                         |
| Host port        | `8000`                                               |
| Container port   | `8000`                                               |
| Cache volume     | `airllm-cache`                                       |
| HuggingFace token| none                                                 |

It then clones the source (if missing), builds the CPU image, starts the container with
the equivalent of:

```sh
docker build -f docker/Dockerfile -t airllm-server:cpu .
docker run -d -p 8000:8000 -v airllm-cache:/cache \
  -e AIRLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct airllm-server:cpu
```

After the health check passes, the interactive chat starts.

### Later runs

An existing `airllm-server` container is detected automatically, so installation is
skipped and the default configuration is reused.

### Chat commands

| Command  | Action                       |
| -------- | ---------------------------- |
| `/help`  | Show available commands      |
| `/clear` | Clear the conversation       |
| `/exit`  | Leave the chat (or `/quit`)  |

`Ctrl+D` or `Ctrl+C` also exits.

## Notes

- The first reply can take minutes: AirLLM downloads the model and splits it into layer
  shards before generating. It reads every layer from disk per token, so throughput is
  seconds per token by design.
- The `airllm-cache` volume holds both the HuggingFace cache and the layer shards.
  Keep it to avoid re-downloading the model on every restart.
- Gated models (for example `meta-llama/*`) require accepting their terms on Hugging
  Face and providing an `HF_TOKEN`.
