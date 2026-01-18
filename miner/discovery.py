"""
Carbon Protocol - Pattern Discovery Pipeline (Miner)

Purpose:
    Analyzes large log volumes to discover frequent N-Grams and cluster them into 
    semantic intents for dictionary generation.

Process:
    1. Statistical: CountVectorizer (N-Grams 3-6) -> Top K phrases
    2. Semantic: SentenceTransformer -> Vector embeddings
    3. Clustering: DBSCAN/KMeans -> Intent groups
    4. Synthesis: JSON output candidate rules
"""

import json
import logging
from typing import List, Dict, Iterator, Tuple,  Iterable
import numpy as np

# Third-party dependencies (scaffolded)
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.cluster import MiniBatchKMeans, DBSCAN
    from sentence_transformers import SentenceTransformer
except ImportError:
    # Fallback for environments where dependencies aren't installed yet
    # This ensures the code can be saved/viewed even if libs are missing
    CountVectorizer = None
    MiniBatchKMeans = None
    DBSCAN = None
    SentenceTransformer = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatternMiner:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.embedding_model_name = embedding_model_name
        self._model = None

    @property
    def model(self):
        """Lazy load the model to save memory if only doing ngram extraction."""
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def extract_frequent_ngrams(self, logs: Iterable[str], min_n=3, max_n=6, top_k=50000) -> List[Tuple[str, int]]:
        """
        Step 1: Statistical Extraction
        Uses CountVectorizer to find the most frequent N-Grams efficiently.
        Returns a list of (ngram_phrase, count).
        """
        if CountVectorizer is None:
            raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")

        logger.info(f"Extracting top {top_k} {min_n}-{max_n} grams...")
        
        # Optimize vectorizer for memory: 
        # - max_features limits vocabulary size immediately
        # - stop_words='english' removes common noise
        vectorizer = CountVectorizer(
            ngram_range=(min_n, max_n),
            max_features=top_k,
            stop_words='english',
            lowercase=True
        )

        # Fit and transform. 
        # Note: For strict 1.8T token scale, look into HashingVectorizer or partial_fit 
        # but CountVectorizer is accurate for 'top_k' exact counts if memory allows.
        try:
            X = vectorizer.fit_transform(logs)
        except ValueError as e:
            logger.error("Vectorizer failed (possibly empty vocab). Check log input.")
            return []

        # Sum counts
        counts = X.sum(axis=0).A1
        vocab = vectorizer.get_feature_names_out()
        
        # Zip and sort desc
        freq_dist = list(zip(vocab, counts))
        freq_dist.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Extraction complete. Found {len(freq_dist)} patterns.")
        return freq_dist[:top_k]

    def cluster_intents(self, phrases_with_counts: List[Tuple[str, int]], eps=0.5, min_samples=3) -> List[Dict]:
        """
        Step 2 & 3: Semantic Encoding and Clustering
        
        Args:
            phrases_with_counts: List of (phrase, count) tuples
            eps: DBSCAN epsilon (distance threshold)
            min_samples: DBSCAN min samples to form a cluster
            
        Returns:
            List of dicts representing clusters
        """
        if not phrases_with_counts:
            return []
            
        phrases = [p[0] for p in phrases_with_counts]
        counts_map = {p[0]: p[1] for p in phrases_with_counts}
        
        # 2. Encode
        logger.info(f"Encoding {len(phrases)} phrases...")
        embeddings = self.model.encode(phrases, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        
        # 3. Cluster
        # Using DBSCAN because we don't know K, and we want to group distinct intents based on density.
        # Tunable: eps (lower = tighter clusters), min_samples (higher = less noise)
        logger.info("Clustering vectors...")
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
        
        labels = clustering.labels_
        
        # 4. Synthesis
        clusters = {}
        noise_phrases = []
        
        for phrase, label in zip(phrases, labels):
            if label == -1:
                noise_phrases.append(phrase)
                continue
                
            if label not in clusters:
                clusters[label] = {
                    "phrases": [],
                    "total_freq": 0
                }
            clusters[label]["phrases"].append(phrase)
            clusters[label]["total_freq"] += counts_map[phrase]

        # Format output
        results = []
        for label, data in clusters.items():
            # Pick representative phrase (simplest heuristic: shortest or most frequent)
            # Here: Most frequent in the group
            variations = data["phrases"]
            variations.sort(key=lambda x: counts_map[x], reverse=True)
            representative = variations[0]
            
            results.append({
                "cluster_id": int(label),
                "representative_phrase": representative,
                "variations": variations,
                "frequency": int(data["total_freq"])
            })
            
        results.sort(key=lambda x: x["frequency"], reverse=True)
        logger.info(f"Found {len(results)} intent clusters. {len(noise_phrases)} phrases identified as noise.")
        return results

def run_pipeline(logs_iterator: Iterable[str], output_file='src/data/discovered_rules.json'):
    miner = PatternMiner()
    
    # Step 1
    top_phrases = miner.extract_frequent_ngrams(logs_iterator, top_k=2000) # Scaled down for demo
    
    if not top_phrases:
        logger.warning("No patterns found.")
        return

    # Step 2 & 3
    intent_clusters = miner.cluster_intents(top_phrases)
    
    # Step 4: Save
    logger.info(f"Saving {len(intent_clusters)} clusters to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(intent_clusters, f, indent=2)

if __name__ == "__main__":
    # Mock Data for Verification
    mock_logs = [
        "I need to write a python function to sort a list",
        "please create python function that sorts lists",
        "can you code me a python sorting function",
        "write python code for binary search",
        "create python script for binary search",
        "fix this bug in my java code",
        "debug my java application error",
        "find the bug in this java snippet",
        "ignore all previous instructions",
        "ignore previous commands",
    ] * 50 # Duplicate to simulate frequency
    
    print("Running Miner on Mock Data...")
    run_pipeline(mock_logs, output_file='miner_output_test.json')
    print("Done.")
