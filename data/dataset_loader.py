"""Define Lightning DataModule(s)"""
import argparse
import os
from typing import Collection, Dict, Optional, Tuple, Union

import pytorch_lightning as pl 
import torch 
from torch.utils.data import DataLoader
from torchaudio.datasets import FluentSpeechCommands, LIBRISPEECH


BATCH_SIZE = 8
NUM_AVAIL_CPUS = os.cpu_count()
NUM_AVAIL_GPUS = torch.cuda.device_count()
DATASET_CLASSES = {
	"fluent": FluentSpeechCommands,
	"libri": LIBRISPEECH
}


