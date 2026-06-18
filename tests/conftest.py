import sys
import os

# Make project root importable when pytest is invoked from the tests/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
