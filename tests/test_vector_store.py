from app.retrieval.vector_store import LocalVectorStore, VectorDocument


def test_local_vector_store_retrieves_semantic_support_example() -> None:
    store = LocalVectorStore(
        [
            VectorDocument(
                doc_id="refund-delay",
                text="customer asks for refund because delivery is delayed and package arrived late",
                metadata={"intent": "refund"},
            ),
            VectorDocument(
                doc_id="invoice-tax",
                text="customer asks for invoice with tax id and billing email",
                metadata={"intent": "invoice"},
            ),
        ]
    )

    hits = store.search("late delivery refund request", k=1)

    assert hits[0].doc_id == "refund-delay"
    assert hits[0].metadata["intent"] == "refund"
    assert hits[0].score > 0


def test_local_vector_store_respects_candidate_filter() -> None:
    store = LocalVectorStore(
        [
            VectorDocument(
                doc_id="refund-delay",
                text="customer asks for refund because delivery is delayed",
                metadata={"intent": "refund"},
            ),
            VectorDocument(
                doc_id="invoice-tax",
                text="customer asks for invoice with tax id and billing email",
                metadata={"intent": "invoice"},
            ),
        ]
    )

    hits = store.search("late delivery refund request", k=2, candidate_ids={"invoice-tax"})

    assert [hit.doc_id for hit in hits] == []
