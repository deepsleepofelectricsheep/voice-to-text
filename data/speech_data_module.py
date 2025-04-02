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
NUM_WORKERS = 0
DATASET_NAME = "fluent"
DATASET_CLASSES = {
	"fluent": {
		"class": FluentSpeechCommands, 
		"train_subset": "train", 
		"val_subset": "valid", 
		"test_subset": "test"
	},
	"libri": {
		"class": LIBRISPEECH, 
		"train_subset": "train-clean-100", 
		"val_subset": "dev-clean", 
		"test_subset": "test-clean"
	},
}
DATASET_DIR = "/data"


class SpeechDataModule(pl.LightningDataModule):
	"""Base Lightning Data Module for speech-to-text datasets"""

	def __init__(self, args: argparse.Namespace = None) -> None:
		super().__init__()
		self.args = vars(args) if args is not None else {}
		self.batch_size = self.args.get("batch_size", BATCH_SIZE)
		self.num_workers = self.args.get("num_workers", NUM_WORKERS)
		self.dataset_dir = self.args.get("dataset_dir", DATASET_DIR)
		self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

		self.dataset_name = self.args.get("dataset_name", DATASET_NAME)
		if self.dataset_name not in DATASET_CLASSES:
			raise ValueError(f"Unsupported dataset: {self.dataset_name}")

		self.dataset_class = DATASET_CLASSES[self.dataset_name]["class"]
		self.dataset_train_subset = DATASET_CLASSES[self.dataset_name]["train_subset"]
		self.dataset_val_subset = DATASET_CLASSES[self.dataset_name]["val_subset"]
		self.dataset_test_subset = DATASET_CLASSES[self.dataset_name]["test_subset"]

	@staticmethod
	def add_to_argparse(parser):
		parser.add_argument(
			"--batch_size",
			type=int,
			default=BATCH_SIZE,
			help=f"Number of examples per forward step. Default is {BATCH_SIZE}",
		)
		parser.add_argument(
		    "--dataset_name",
		    type=str,
		    default=DATASET_NAME,
		    help=(
          		"Name of dataset to download from torchaudio. "
		        "Available options: 'fluent', 'libri'. "
		        f"Default is '{DATASET_NAME}'."
		    ),
		)

	def setup(self, stage=None):
		if self.dataset_name == "fluent":
			self.train_dataset = self.dataset_class(self.dataset_dir, subset=self.dataset_train_subset)
			self.val_dataset = self.dataset_class(self.dataset_dir, subset=self.dataset_val_subset)
			self.test_dataset = self.dataset_class(self.dataset_dir, subset=self.dataset_test_subset)
		elif self.dataset_name == "libri":
			self.train_dataset = self.dataset_class(self.dataset_dir, url=self.dataset_train_subset, download=True)
			self.val_dataset = self.dataset_class(self.dataset_dir, url=self.dataset_val_subset, download=True)
			self.test_dataset = self.dataset_class(self.dataset_dir, url=self.dataset_test_subset, download=True)

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset, 
			batch_size=self.batch_size, 
			shuffle=True, 
			num_workers=self.num_workers
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset, 
			batch_size=self.batch_size, 
			shuffle=True, 
			num_workers=self.num_workers
		)		

	def test_dataloader(self):
		return DataLoader(
			self.test_dataset, 
			batch_size=self.batch_size, 
			shuffle=True, 
			num_workers=self.num_workers
		)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	SpeechDataModule.add_to_argparse(parser)
	args = parser.parse_args()
	dm = SpeechDataModule(args)
	dm.setup()