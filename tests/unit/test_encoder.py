import numpy as np
from nexus.core.unified_encoder import UnifiedEncoder, Modality

def test_unified_encoder_fallback():
    encoder = UnifiedEncoder({"text_model": "invalid-model"})
    
    # Test encoding text
    vec1 = encoder.encode("Hello world", Modality.TEXT)
    assert vec1.shape == (1024,)
    assert np.isclose(np.linalg.norm(vec1), 1.0, atol=1e-5)
    
    # Test fallback determinism
    vec2 = encoder.encode("Hello world", Modality.TEXT)
    assert np.allclose(vec1, vec2)
    
    # Test different modality (should still use fallback and output 1024-dim)
    vec3 = encoder.encode("print('Hello')", Modality.CODE)
    assert vec3.shape == (1024,)
    
def test_batch_encode():
    encoder = UnifiedEncoder({"text_model": "invalid-model"})
    items = [
        ("Hello", Modality.TEXT),
        ("World", Modality.TEXT),
        ("def foo(): pass", Modality.CODE)
    ]
    
    results = encoder.batch_encode(items, batch_size=2)
    assert len(results) == 3
    for res in results:
        assert res.shape == (1024,)
        assert np.isclose(np.linalg.norm(res), 1.0, atol=1e-5)
