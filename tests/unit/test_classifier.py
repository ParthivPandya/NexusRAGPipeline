from nexus.core.query_classifier import QueryDNAClassifier

def test_query_classifier_factual():
    classifier = QueryDNAClassifier()
    dna = classifier.classify("What is the capital of France?")
    
    assert dna.factual > 0.5
    assert dna.dominant_dimension == "factual"
    assert dna.retrieval_strategy["dense_hnsw"] > 0
    assert dna.retrieval_strategy["bm25_sparse"] > 0
    
def test_query_classifier_temporal():
    classifier = QueryDNAClassifier()
    dna = classifier.classify("What was Tesla's revenue in 2022?")
    
    assert dna.temporal > 0.3
    assert "temporal_index" in dna.retrieval_strategy
    assert dna.retrieval_strategy["temporal_index"] > 0.1
    
def test_query_classifier_comparative():
    classifier = QueryDNAClassifier()
    dna = classifier.classify("React vs Angular performance")
    
    assert dna.comparative > 0.4
    assert dna.dominant_dimension == "comparative"
    assert dna.retrieval_strategy["multi_branch"] > 0.1
