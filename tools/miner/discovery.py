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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

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
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def extract_frequent_ngrams(self, logs: Iterable[str], min_n=3, max_n=6, top_k=50000) -> List[Tuple[str, int]]:
        """
        Step 1: Statistical Extraction
        Uses CountVectorizer to find the most frequent N-Grams efficiently.
        Returns a list of (ngram_phrase, count).
        """
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

import argparse

def run_pipeline(logs_iterator: Iterable[str], output_file='src/data/discovered_rules.json', top_k=2000):
    miner = PatternMiner()
    
    # Step 1
    top_phrases = miner.extract_frequent_ngrams(logs_iterator, top_k=top_k) 
    
    if not top_phrases:
        logger.warning("No patterns found.")
        return

    # Step 2 & 3
    intent_clusters = miner.cluster_intents(top_phrases)
    
    # Step 4: Save
    logger.info(f"Saving {len(intent_clusters)} clusters to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(intent_clusters, f, indent=2)

def load_logs_from_file(filepath: str) -> Iterator[str]:
    """Generator to yield lines from a large log file without storing in memory."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    except FileNotFoundError:
        logger.error(f"Log file not found: {filepath}")
        return

def load_logs_from_hf(dataset_name: str, split='train', max_samples=None, token=None) -> Iterator[str]:
    """Generator to yield prompts from a HuggingFace dataset (Streaming Mode)."""
    logger.info(f"Streaming dataset: {dataset_name} (split={split})...")
    # Streaming=True allows processing 1TB+ datasets without full download
    ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
    
    count = 0
    log_interval = max(1, (max_samples or 10000) // 20)  # Log every 5%
    
    for row in ds:
        # Heuristic to find the 'text' content
        text = None
        
        # 1. Standard text column
        if 'text' in row:
            text = row['text']
        # 2. Chatbot Arena / Conversation format
        elif 'conversation_a' in row:
             # Extract user messages from conversation
             text = " ".join([turn['content'] for turn in row['conversation_a'] if turn['role'] == 'user'])
        elif 'conversations' in row:
             text = " ".join([turn['value'] for turn in row['conversations'] if turn['from'] == 'human'])
        # 3. Prompt column
        elif 'prompt' in row:
             text = row['prompt']
        # 4. ShareGPT / OpenAssistant style
        elif 'messages' in row:
             text = " ".join([m['content'] for m in row['messages'] if m.get('role') == 'user'])
        # 5. Instruction style
        elif 'instruction' in row:
             text = row['instruction']
             
        if text:
            # Flatten newlines for log-like processing
            clean_text = text.replace('\n', ' ').strip()
            if clean_text:
                yield clean_text
                count += 1
                
                # Progress logging
                if count % log_interval == 0:
                    if max_samples:
                        pct = (count / max_samples) * 100
                        print(f"\r  Progress: {count:,}/{max_samples:,} samples ({pct:.1f}%)", end="", flush=True)
                    else:
                        print(f"\r  Progress: {count:,} samples processed", end="", flush=True)
                
        if max_samples and count >= max_samples:
            print()  # Newline after progress
            logger.info(f"Reached max_samples limit: {max_samples}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carbon Protocol Pattern Miner")
    parser.add_argument("--input", type=str, help="Path to input log file (txt/log)", required=False)
    parser.add_argument("--hf-dataset", type=str, help="HuggingFace dataset name (e.g. lmsys/chatbot_arena_conversations)", required=False)
    parser.add_argument("--hf-token", type=str, help="HuggingFace API token for gated datasets", required=False)
    parser.add_argument("--output", type=str, default="miner_output.json", help="Path to output JSON")
    parser.add_argument("--top-k", type=int, default=50000, help="Number of top N-Grams to analyze")
    parser.add_argument("--limit", type=int, default=10000, help="Max samples to process from HF stream")
    
    args = parser.parse_args()

    if args.input:
        print(f"Running Miner on Local File: {args.input}")
        logs_gen = load_logs_from_file(args.input)
        run_pipeline(logs_gen, output_file=args.output, top_k=args.top_k)
    elif args.hf_dataset:
        print(f"Running Miner on HuggingFace Dataset: {args.hf_dataset}")
        logs_gen = load_logs_from_hf(args.hf_dataset, max_samples=args.limit, token=args.hf_token)
        run_pipeline(logs_gen, output_file=args.output, top_k=args.top_k)
    else:
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
    
        print("Running Miner on Mock Data (No input file provided)...")
        print("Usage: python miner/discovery.py --input <file.txt> OR --hf-dataset <name>")
        run_pipeline(mock_logs, output_file='miner_output_test.json', top_k=50)
        print("Done.")
