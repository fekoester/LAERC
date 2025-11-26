#!/usr/bin/env python3
"""
Deprecated shim: call prepare_gpt2_tokens.py with dataset=custom by default.
"""

from scripts.prepare_gpt2_tokens import main

if __name__ == "__main__":
    main()
