# ND Factor Core — Obfuscated Black Box

def compute_narrative_divergence(texts):
    """
    Compute Narrative Dispersion from a list of text strings.
    Returns: ND (float), std (float), similarity_list (list)
    
    IMPLEMENTATION NOTE:
    This is a black-box wrapper. The actual embedding model and
    similarity computation methodology are proprietary.
    
    Input:
        texts: list of Chinese-language strings (post titles)
    Returns:
        nd: 1 - mean(pairwise_similarity)  [0, 1]
            0 = all texts identical → maximum narrative convergence
            1 = all texts orthogonal → maximum narrative dispersion
    """
    from ._engine import get_embedder
    model = get_embedder()
    return _compute(texts, model)

def _compute(texts, model):
    """
    Internal computation. Uses proprietary text representation and
    similarity aggregation methodology.
    """
    import numpy as np
    
    if len(texts) < 5:
        return 0.5, 0.0, []
    
    # Embed all texts
    embeddings = model.encode(texts)
    
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    
    # Pairwise cosine similarity (upper triangle)
    n = len(embeddings)
    sim_matrix = embeddings @ embeddings.T
    iu = np.triu_indices(n, k=1)
    similarities = sim_matrix[iu]
    
    nd = float(1.0 - np.mean(similarities))
    nd_std = float(np.std(similarities))
    
    return nd, nd_std, similarities.tolist()
