# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Chat Completions behavior when persistence is disabled.

When an operator configures ``storage.stores.inference`` as ``null``, OGX must
not construct an ``InferenceStore`` at all: no ``inference_store`` table is
created, no background write workers run, and no chat completion request or
response payload is ever persisted. Chat completions still work for both
streaming and non-streaming requests, and the history endpoints report that
persistence is not configured (501) rather than returning an empty list or 404.

This test builds a ``StackConfig`` from the ``ci-tests`` distribution with the
inference store reference removed, boots an in-process library client from it,
and exercises the full HTTP path through the OpenAI-compatible client.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from ogx.core.datatypes import StackConfig
from ogx.core.library_client import OGXAsLibraryClient
from ogx.core.stack import get_stack_run_config_from_distro
from tests.integration.inference.store_disabled_constants import (
    NON_STREAMING_PROMPT,
    STREAMING_PROMPT,
    TEXT_MODEL,
)


def _build_non_persisting_config(sqlite_dir: str) -> tuple[StackConfig, Path]:
    run_config = get_stack_run_config_from_distro("ci-tests")
    # Disable chat completion persistence: no store is constructed, no table is
    # created, and no background write workers are started.
    run_config.storage.stores.inference = None
    # Vector-store config pulls in embedding/reranker model validation that is
    # unrelated to chat completion persistence and makes every in-process boot
    # load the sentence-transformers stack; drop it so booting is deterministic.
    run_config.vector_stores = None
    return run_config, Path(sqlite_dir) / "sql_store.db"


@pytest.fixture(scope="session")
def non_persisting_client():
    """Boot an in-process library client with chat completion persistence disabled.

    Booted once for the whole session because constructing the ci-tests stack is
    expensive. These tests do not mutate shared client state, so a single shared
    client is safe.

    This deliberately does not use the shared ``ogx_client`` fixture: unlike that
    fixture it must boot its own stack with a custom store config
    (``inference: null``), which cannot be expressed through the standard
    server-mode run config. See ``_boot_library_client`` for how the in-process
    boot avoids colliding with the outer server-mode OGX server.
    """
    # The OpenAI provider validates that an API key is present before the
    # recording harness short-circuits the request in replay mode. A fake key
    # is sufficient because the request is never sent to the provider.
    os.environ["OPENAI_API_KEY"] = "fake-key-for-replay"
    sqlite_dir = tempfile.mkdtemp(prefix="ogx-no-store-")
    os.environ["SQLITE_STORE_DIR"] = sqlite_dir
    run_config, sql_db = _build_non_persisting_config(sqlite_dir)
    config_file = tempfile.NamedTemporaryFile(delete=False, suffix="-run.yaml").name
    with open(config_file, "w") as f:
        yaml.dump(run_config.model_dump(mode="json"), f)
    client = _boot_library_client(config_file)
    try:
        yield client, sql_db
    finally:
        client.shutdown()
        os.unlink(config_file)


def _boot_library_client(config_file: str) -> OGXAsLibraryClient:
    """Boot an in-process library client with the standalone metrics endpoint off.

    ``integration-tests.sh`` exports ``OGX_METRICS_ENDPOINT_ENABLED=1`` so the outer
    server-mode OGX server exposes a metrics scrape endpoint on port 9464. Booting a
    second stack in-process would try to bind the same port; nothing scrapes the
    in-process stack, so the endpoint is turned off for the boot and restored right
    after so sibling tests in the same pytest process (e.g. the metrics endpoint
    integration test) still observe the script's flag.
    """
    metrics_env = os.environ.get("OGX_METRICS_ENDPOINT_ENABLED")
    os.environ["OGX_METRICS_ENDPOINT_ENABLED"] = "0"
    try:
        return OGXAsLibraryClient(config_file, skip_logger_removal=True)
    finally:
        if metrics_env is not None:
            os.environ["OGX_METRICS_ENDPOINT_ENABLED"] = metrics_env
        else:
            os.environ.pop("OGX_METRICS_ENDPOINT_ENABLED", None)


def test_non_streaming_chat_completion_without_store(non_persisting_client):
    """A non-streaming completion is returned normally with id/model populated."""
    client, _ = non_persisting_client
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": NON_STREAMING_PROMPT}],
    )
    assert response.id
    assert response.model == TEXT_MODEL
    assert response.choices
    assert response.choices[0].message.content


def test_streaming_chat_completion_without_store(non_persisting_client):
    """A streaming completion streams normally with the requested model id."""
    client, _ = non_persisting_client
    stream = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": STREAMING_PROMPT}],
        stream=True,
    )
    chunks = list(stream)
    assert chunks
    response_id = None
    for chunk in chunks:
        assert chunk.model == TEXT_MODEL
        if chunk.id:
            response_id = chunk.id
    assert response_id


def test_list_chat_completions_reports_not_configured(non_persisting_client):
    """list raises a not-configured error rather than returning an empty list.

    In library-client mode the router's ``NotImplementedError`` propagates
    directly; the server's exception mapping translates it to HTTP 501.
    """
    client, _ = non_persisting_client
    with pytest.raises(NotImplementedError):
        client.chat.completions.list(limit=10)


def test_retrieve_chat_completion_reports_not_configured(non_persisting_client):
    """retrieve for a just-completed id raises a not-configured error rather than a 404."""
    client, _ = non_persisting_client
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": NON_STREAMING_PROMPT}],
    )
    with pytest.raises(NotImplementedError):
        client.chat.completions.retrieve(response.id)


def test_list_chat_completion_messages_reports_not_configured(non_persisting_client):
    """messages raises a not-configured error, consistent with list/retrieve."""
    client, _ = non_persisting_client
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": NON_STREAMING_PROMPT}],
    )
    with pytest.raises(NotImplementedError):
        client.chat.completions.messages.list(completion_id=response.id)


def test_no_inference_store_table_when_persistence_disabled(non_persisting_client):
    """No ``inference_store`` table exists in the SQL backend, proving payloads were never written."""
    _client, sql_db = non_persisting_client
    if not Path(sql_db).exists():
        pytest.skip(f"SQL backend db not found at {sql_db}")
    import sqlite3

    conn = sqlite3.connect(str(sql_db))
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='inference_store'").fetchall()
    finally:
        conn.close()
    assert rows == [], "inference_store table must not exist when persistence is disabled"
