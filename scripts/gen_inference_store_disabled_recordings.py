#!/usr/bin/env python3
"""Generate recordings for test_inference_store_disabled.py against a local mock OpenAI server.

Run from repo root:
    uv run python scripts/gen_inference_store_disabled_recordings.py

This spins up a tiny OpenAI-compatible HTTP mock, boots the ci-tests stack with
the inference store disabled and openai pointed at the mock, and exercises the
same chat-completion calls the test makes -- in RECORD mode -- so the recording
harness captures them into tests/integration/inference/recordings/.

A mock is used instead of a live provider because the test only needs the
id/model populated and asserts nothing about real completion content; this
keeps the recordings self-contained and regenerable offline.
"""

import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from ogx.core.library_client import OGXAsLibraryClient  # noqa: E402
from ogx.core.stack import get_stack_run_config_from_distro  # noqa: E402
from ogx.core.testing_context import set_test_context  # noqa: E402

RECORDINGS_DIR = os.path.join(
    REPO_ROOT, "tests", "integration", "inference", "recordings"
)

# The exact prompts the test uses -- the recording hash depends on the body.
TEXT_MODEL = "openai/gpt-4o"
NON_STREAMING_PROMPT = "Say hello."
STREAMING_PROMPT = "Say hello in one sentence."

# pytest node ids the test will use -- the recording hash includes the test id.
TEST_NODE_IDS = [
    "tests/integration/inference/test_inference_store_disabled.py::test_non_streaming_chat_completion_without_store",
    "tests/integration/inference/test_inference_store_disabled.py::test_streaming_chat_completion_without_store",
    "tests/integration/inference/test_inference_store_disabled.py::test_retrieve_chat_completion_reports_not_configured",
    "tests/integration/inference/test_inference_store_disabled.py::test_list_chat_completion_messages_reports_not_configured",
]

MOCK_HOST = "127.0.0.1"
MOCK_PORT = 0  # ephemeral


def _completion_body(model: str, prompt: str, completion_id: str) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"Response to: {prompt}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


def _stream_chunks(model: str, completion_id: str):
    return [
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": " world."}, "finish_reason": None}
            ],
        },
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]


class MockHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith("/models"):
            body = json.dumps(
                {"object": "list", "data": [{"id": "gpt-4o", "object": "model"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}
        stream = payload.get("stream", False)
        model = payload.get("model", "gpt-4o")
        completion_id = "chatcmpl-mock-recording"
        if stream:
            chunks = _stream_chunks(model, completion_id)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for ch in chunks:
                self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            body = _completion_body(model, "Say hello.", completion_id)
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def log_message(self, *args):  # silence
        pass


def start_mock_server() -> HTTPServer:
    server = HTTPServer((MOCK_HOST, MOCK_PORT), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


async def run(test_id: str, config_path: str) -> None:
    set_test_context(test_id)
    client = OGXAsLibraryClient(config_path, skip_logger_removal=True)
    try:
        if test_id.endswith("::test_streaming_chat_completion_without_store"):
            stream = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[{"role": "user", "content": STREAMING_PROMPT}],
                stream=True,
            )
            list(stream)
        else:
            client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[{"role": "user", "content": NON_STREAMING_PROMPT}],
            )
    finally:
        client.shutdown()


def main() -> None:
    os.environ["OGX_TEST_INFERENCE_MODE"] = "record"
    os.environ["OGX_LOGGING"] = "all=warning"
    os.environ["OPENAI_API_KEY"] = "fake-key-for-replay"
    sqlite_dir = tempfile.mkdtemp(prefix="ogx-record-")
    os.environ["SQLITE_STORE_DIR"] = sqlite_dir

    server = start_mock_server()
    port = server.server_address[1]
    mock_base = f"http://{MOCK_HOST}:{port}/v1"
    os.environ["OPENAI_BASE_URL"] = mock_base

    run_config = get_stack_run_config_from_distro("ci-tests")
    run_config.storage.stores.inference = None
    run_config.vector_stores = None

    config_file = os.path.join(tempfile.mkdtemp(), "run.yaml")
    with open(config_file, "w") as f:
        yaml.dump(run_config.model_dump(mode="json"), f)

    # Isolate recording so the shared inference recordings directory is untouched.
    # The recorder always writes into the test file's ``recordings/`` dir (relative
    # to CWD) when a test context is set, so move the real directory aside while
    # recording and merge only the chat-completion recordings back afterwards.
    backup = None
    if os.path.isdir(RECORDINGS_DIR):
        backup = RECORDINGS_DIR + ".bak"
        shutil.move(RECORDINGS_DIR, backup)

    try:
        for test_id in TEST_NODE_IDS:
            print(f"recording for {test_id} ...")
            asyncio.run(run(test_id, config_file))
    finally:
        server.shutdown()

    # Collect the freshly recorded chat-completion recordings (skip models-list
    # recordings -- the test does not need them in replay mode) and rewrite their
    # mock-host URLs to the canonical provider URL. The recording hash ignores
    # the host, so replay works against the real provider URL.
    staged = tempfile.mkdtemp(prefix="ogx-staged-")
    for name in os.listdir(RECORDINGS_DIR):
        if not name.endswith(".json") or name.startswith("models-"):
            continue
        src = os.path.join(RECORDINGS_DIR, name)
        with open(src) as f:
            data = json.load(f)
        url = data.get("request", {}).get("url", "")
        if re.search(r"\d+\.\d+\.\d+\.\d+:\d+", url):
            data["request"]["url"] = (
                "https://api.openai.com/v1" + url.split("/v1", 1)[1]
            )
        with open(os.path.join(staged, name), "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    # Restore the original recordings directory and drop in the new recordings.
    shutil.rmtree(RECORDINGS_DIR, ignore_errors=True)
    if backup is not None:
        shutil.move(backup, RECORDINGS_DIR)
    else:
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
    n_written = len(os.listdir(staged))
    for name in os.listdir(staged):
        shutil.copy2(os.path.join(staged, name), os.path.join(RECORDINGS_DIR, name))
    shutil.rmtree(staged, ignore_errors=True)
    print(f"wrote {n_written} recordings")
    print("done")


if __name__ == "__main__":
    main()
