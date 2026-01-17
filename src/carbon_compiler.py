import re

class CarbonCompiler:
    """
    Reference implementation of the Carbon Protocol Compiler (v1.0).
    Optimizes natural language prompts into deterministic command syntax.
    """
    
    def __init__(self, ruleset="v1"):
        self.ruleset = ruleset
        # Domain Registry: Maps intent to Carbon Syntax
        self.registry = {
            "python": "@PY",
            "script": "CMD:CODE",
            "scrape": "ACT:SCRAPE",
            "data": "OBJ:DATA",
            "kubernetes": "CMD:K8S",
            "deployment": "TYPE:DEPLOY",
            "sql": "CMD:SQL",
            "query": "ACT:QUERY",
            "summarize": "CMD:SUMMARIZE",
            "email": "CMD:EMAIL"
        }
        # Stop Words: Linguistic noise to be stripped
        self.stop_words = [
            "please", "could", "you", "write", "a", "an", "the", 
            "to", "from", "for", "is", "are", "can", "i", "need"
        ]

    def compress(self, text):
        """
        Converts raw text input into Carbon Syntax.
        """
        # 1. Normalize
        tokens = text.lower().replace("?", "").replace(".", "").split()
        
        # 2. Filter & Map
        syntax_stream = []
        for token in tokens:
            if token in self.stop_words:
                continue
            
            # Check Registry
            if token in self.registry:
                syntax_stream.append(self.registry[token])
            
        # 3. Assemble (Pipe delimited for Carbon Standard)
        if not syntax_stream:
            return "ERROR: NO_INTENT_DETECTED"
            
        return " | ".join(syntax_stream)

# Test Block
if __name__ == "__main__":
    cc = CarbonCompiler()
    sample = "Could you please write a python script to scrape data"
    print(cc.compress(sample))