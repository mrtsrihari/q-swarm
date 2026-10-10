"""Streamlit Community Cloud Main Application Entrypoint for Q-Swarm RL."""

import os
import sys

# Ensure src/ package is in Python search path for frictionless cloud deployment
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import and execute main dashboard module
import qswarm.dashboard.app  # noqa: E402, F401
