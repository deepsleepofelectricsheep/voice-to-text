"""Experiment-running framework"""
import sys
import os

# Get the absolute path of the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Project root directory: {project_root}")
sys.path.insert(0, project_root)

import argparse
from data.speech_data_module import SpeechDataModule

parser = argparse.ArgumentParser()
SpeechDataModule.add_to_argparse(parser)
args = parser.parse_args()
dm = SpeechDataModule(args)
dm.setup()