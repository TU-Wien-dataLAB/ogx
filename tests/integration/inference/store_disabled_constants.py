# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared constants for the inference-store-disabled tests and their recording generator.

The recording harness keys recordings by a SHA256 hash of the request body and the
pytest node id, so the model id, prompts, and node ids MUST match between
``tests/integration/inference/test_inference_store_disabled.py`` and
``scripts/gen_inference_store_disabled_recordings.py``. Keeping them in one module
prevents silent drift between the test and its regenerable recordings.
"""

TEXT_MODEL = "openai/gpt-4o"

NON_STREAMING_PROMPT = "Say hello."
STREAMING_PROMPT = "Say hello in one sentence."

# The full pytest node ids the generator records completions for. The test's
# list/retrieve/messages tests raise before any provider call, so only the four
# ids below perform chat-completion requests worth recording; the remaining tests
# need no recording.
TEST_MODULE = "tests/integration/inference/test_inference_store_disabled.py"
NON_STREAMING_TEST = f"{TEST_MODULE}::test_non_streaming_chat_completion_without_store"
STREAMING_TEST = f"{TEST_MODULE}::test_streaming_chat_completion_without_store"
RETRIEVE_TEST = f"{TEST_MODULE}::test_retrieve_chat_completion_reports_not_configured"
MESSAGES_TEST = f"{TEST_MODULE}::test_list_chat_completion_messages_reports_not_configured"

RECORDING_TEST_IDS = [NON_STREAMING_TEST, STREAMING_TEST, RETRIEVE_TEST, MESSAGES_TEST]
