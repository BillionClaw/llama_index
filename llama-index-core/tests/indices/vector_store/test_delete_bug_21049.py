"""Test for VectorStoreIndex delete helpers bug - issue #21049."""

import pytest
from typing import Any, List

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.schema import Document
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores.simple import SimpleVectorStore


# Global tracker for delete calls
_delete_calls: List[tuple] = []


class TrackingVectorStore(SimpleVectorStore):
    """Vector store that tracks delete calls."""

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        global _delete_calls
        _delete_calls.append(("SYNC", ref_doc_id))
        super().delete(ref_doc_id, **kwargs)

    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        global _delete_calls
        _delete_calls.append(("ASYNC", ref_doc_id))
        await super().adelete(ref_doc_id, **kwargs)


@pytest.fixture
def mock_embed_model():
    return MockEmbedding(embed_dim=8)


def test_delete_ref_doc_does_not_call_delete_with_node_id(mock_embed_model):
    """
    Test that delete_ref_doc does not incorrectly call delete() with node IDs.

    Regression test for https://github.com/run-llama/llama_index/issues/21049.

    The bug was that _delete_from_index_struct() was calling
    self._vector_store.delete(node_id) with node IDs, but delete() expects
    ref_doc_id. This test verifies that delete() is only called with ref_doc_id.
    """
    global _delete_calls
    _delete_calls.clear()

    store = TrackingVectorStore()
    index = VectorStoreIndex.from_documents(
        [Document(text="hello world", doc_id="my-doc-id")],
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=mock_embed_model,
    )

    # Clear any delete calls from initial setup
    _delete_calls.clear()

    # Delete the document
    index.delete_ref_doc("my-doc-id")

    # Should only have one delete call with the ref_doc_id
    # NOT with any node IDs
    print(f"Delete calls: {_delete_calls}")

    # Check that we only have SYNC calls with ref_doc_id, not node IDs
    sync_calls = [call for call in _delete_calls if call[0] == "SYNC"]
    async_calls = [call for call in _delete_calls if call[0] == "ASYNC"]

    # Should only have one sync call with ref_doc_id
    assert len(sync_calls) == 1, f"Expected 1 sync delete call, got {sync_calls}"
    assert sync_calls[0][1] == "my-doc-id", f"Expected ref_doc_id, got {sync_calls[0][1]}"

    # Should not have any async calls in sync path
    assert len(async_calls) == 0, f"Expected 0 async delete calls, got {async_calls}"


@pytest.mark.asyncio
async def test_adelete_ref_doc_does_not_call_sync_delete(mock_embed_model):
    """
    Test that adelete_ref_doc does not incorrectly call sync delete().

    Regression test for https://github.com/run-llama/llama_index/issues/21049.

    The bug was that _adelete_from_index_struct() was calling
    self._vector_store.delete(node_id) synchronously in an async context,
    and also passing node_id instead of ref_doc_id.
    """
    global _delete_calls
    _delete_calls.clear()

    store = TrackingVectorStore()
    index = VectorStoreIndex.from_documents(
        [Document(text="hello world", doc_id="my-doc-id")],
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=mock_embed_model,
    )

    # Clear any delete calls from initial setup
    _delete_calls.clear()

    # Delete the document asynchronously
    await index.adelete_ref_doc("my-doc-id")

    print(f"Delete calls: {_delete_calls}")

    # Should only have async calls in async path
    sync_calls = [call for call in _delete_calls if call[0] == "SYNC"]
    async_calls = [call for call in _delete_calls if call[0] == "ASYNC"]

    # In async path, we should have async delete calls, not sync
    # The bug caused sync delete calls to be made in async context
    assert len(async_calls) >= 1, f"Expected at least 1 async delete call, got {async_calls}"

    # Check that all async calls are for ref_doc_id, not node IDs
    for call_type, call_id in async_calls:
        assert call_id == "my-doc-id", f"Expected ref_doc_id 'my-doc-id', got '{call_id}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
