"""
Standalone launcher — run from anywhere:
    python run_pipeline.py --sketch_dir ... --hf_token ...
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from KontextPipeline.orchestrator import main
main()
