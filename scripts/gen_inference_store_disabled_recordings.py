#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

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
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

from ogx.core.library_client import OGXAsLibraryClient  # noqa: E402  # Requires local checkout path setup above.
from ogx.core.testing_context import set_test_context  # noqa: E402  # Requires local checkout path setup above.
from tests.integration.inference.store_disabled_support import (  # noqa: E402  # Requires path setup above.
    NON_STREAMING_PROMPT,
    RECORDING_TEST_IDS,
    STREAMING_PROMPT,
    TEXT_MODEL,
    build_inference_store_disabled_run_config,
)

RECORDINGS_DIR = os.path.join(REPO_ROOT, "tests", "integration", "inference", "recordings")

# pytest node ids the test will use -- the recording hash includes the test id.
TEST_NODE_IDS = RECORDING_TEST_IDS

MOCK_HOST = "127.0.0.1"
MOCK_PORT = 0  # ephemeral


def _completion_body(model: str, prompt: str, completion_id: str) -> dict[str, Any]:
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


def _stream_chunks(model: str, completion_id: str) -> list[dict[str, Any]]:
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
            "choices": [{"index": 0, "delta": {"content": " world."}, "finish_reason": None}],
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
    def do_GET(self) -> None:  # noqa: N802 Function name `do_GET` should be lowercase
        if self.path.endswith("/models"):
            body = json.dumps({"object": "list", "data": [{"id": "gpt-4o", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 Function name `do_POST` should be lowercase
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            decoded_payload = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            decoded_payload = {}
        payload = decoded_payload if isinstance(decoded_payload, dict) else {}
        stream = payload.get("stream", False)
        model = payload.get("model", "gpt-4o")
        completion_id = "chatcmpl-mock-recording"
        if stream:
            chunks = _stream_chunks(model, completion_id)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            body = _completion_body(model, NON_STREAMING_PROMPT, completion_id)
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # silence
        pass


@contextmanager
def _mock_server() -> Iterator[HTTPServer]:
    server = HTTPServer((MOCK_HOST, MOCK_PORT), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def _run(test_id: str, config_path: str) -> None:
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


def _stage_recordings(staged_dir: str) -> int:
    """Copy generated chat recordings to staging and normalize their provider URLs."""
    for name in os.listdir(RECORDINGS_DIR):
        if not name.endswith(".json") or name.startswith("models-"):
            continue
        source = os.path.join(RECORDINGS_DIR, name)
        with open(source, encoding="utf-8") as file:
            data = json.load(file)
        url = data.get("request", {}).get("url", "")
        if re.search(r"\d+\.\d+\.\d+\.\d+:\d+", url):
            data["request"]["url"] = "https://api.openai.com/v1" + url.split("/v1", 1)[1]
        with open(os.path.join(staged_dir, name), "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
    return len(os.listdir(staged_dir))


def _generate_recordings(config_file: str) -> int:
    """Generate recordings while preserving the repository's existing fixtures."""
    recordings_parent = os.path.dirname(RECORDINGS_DIR)
    with (
        tempfile.TemporaryDirectory(prefix=".recordings-backup-", dir=recordings_parent) as backup_dir,
        tempfile.TemporaryDirectory(prefix="ogx-staged-") as staged_dir,
    ):
        original_recordings = os.path.join(backup_dir, "recordings")
        had_original_recordings = os.path.isdir(RECORDINGS_DIR)
        if had_original_recordings:
            shutil.move(RECORDINGS_DIR, original_recordings)

        try:
            for test_id in TEST_NODE_IDS:
                print(f"recording for {test_id} ...")
                asyncio.run(_run(test_id, config_file))
            n_written = _stage_recordings(staged_dir)
        finally:
            shutil.rmtree(RECORDINGS_DIR, ignore_errors=True)
            if had_original_recordings:
                shutil.move(original_recordings, RECORDINGS_DIR)

        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        for name in os.listdir(staged_dir):
            shutil.copy2(os.path.join(staged_dir, name), os.path.join(RECORDINGS_DIR, name))
        return n_written


def main() -> None:
    os.environ["OGX_TEST_INFERENCE_MODE"] = "record"
    os.environ["OGX_LOGGING"] = "all=warning"
    os.environ["OPENAI_API_KEY"] = "fake-key-for-replay"

    with (
        tempfile.TemporaryDirectory(prefix="ogx-record-") as sqlite_dir,
        tempfile.TemporaryDirectory(prefix="ogx-config-") as config_dir,
        _mock_server() as server,
    ):
        os.environ["SQLITE_STORE_DIR"] = sqlite_dir
        port = server.server_address[1]
        os.environ["OPENAI_BASE_URL"] = f"http://{MOCK_HOST}:{port}/v1"

        run_config = build_inference_store_disabled_run_config()
        config_file = os.path.join(config_dir, "run.yaml")
        with open(config_file, "w", encoding="utf-8") as file:
            yaml.safe_dump(run_config.model_dump(mode="json"), file)

        n_written = _generate_recordings(config_file)

    print(f"wrote {n_written} recordings")
    print("done")


if __name__ == "__main__":
    main()
