#!/usr/bin/env python
"""
predict_image.py
----------------
Convenience script for image prediction with explainability.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predict import main

if __name__ == "__main__":
    main()

