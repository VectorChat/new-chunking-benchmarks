# first line: 83
@joblib.Memory(location='./cachedir_chroma_1nov', verbose=0).cache(ignore=['embedder', 'db_client'])
def retrieve_relevant_chunks(
    question: str,
    db: VectorDB,
    embedder,
    db_client,
    index_name: str,
    namespace: str,
    retriever_topk=1000,
    rerank_topk=3,
    filter={},
    **kwargs,
) -> List[str]:
    # gen embedding for the question
    logging.info(f"retriever_topk {retriever_topk}, rerank_topk {rerank_topk}")
    question_embedding = embedder.embed_chunk(question)

    # retrieve topk chunks
    vectors = db_client.query(
        index_name, vector=question_embedding, top_k=retriever_topk, filter=filter, namespace=namespace
    )
    # print(f'retrieved metadata: {[x["metadata"]["chapter_name"] for x in vectors["matches"]]}')

    # rerank
    # reranker = CohereReranker()
    reranker = AWSReranker()
    reranked_docs = reranker.rerank(
        query=question,
        res=vectors,
        top_n=rerank_topk,
    )
    reranked_docs = [clean_text(doc) for doc in reranked_docs]
    return reranked_docs
