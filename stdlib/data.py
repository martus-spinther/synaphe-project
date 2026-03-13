"""
Synaphe Data Pipeline Standard Library v0.3.0
The Synaphe Project

Functions for loading, validating, and transforming data.
Bridges the gap between raw files and typed Synaphe tensors.
"""

import math
import random
import struct
import gzip
import os
from typing import List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field


# ── Try importing backends ───────────────────────────────────────────

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════
# SCHEMA VALIDATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SchemaField:
    name: str
    dtype: str  # "int", "float", "string", "tensor"
    shape: Optional[Tuple[int, ...]] = None
    constraints: List[Callable] = field(default_factory=list)
    nullable: bool = False


@dataclass
class Schema:
    """Data validation schema — checked at data boundaries."""
    name: str
    fields: List[SchemaField] = field(default_factory=list)

    def validate(self, record: dict) -> Tuple[bool, List[str]]:
        errors = []
        for f in self.fields:
            if f.name not in record:
                if not f.nullable:
                    errors.append(f"Missing required field: {f.name}")
                continue

            value = record[f.name]

            # Type check
            if f.dtype == "int" and not isinstance(value, int):
                errors.append(f"{f.name}: expected int, got {type(value).__name__}")
            elif f.dtype == "float" and not isinstance(value, (int, float)):
                errors.append(f"{f.name}: expected float, got {type(value).__name__}")
            elif f.dtype == "string" and not isinstance(value, str):
                errors.append(f"{f.name}: expected string, got {type(value).__name__}")

            # Shape check for tensors
            if f.shape and hasattr(value, 'shape'):
                actual_shape = tuple(value.shape)
                if actual_shape != f.shape:
                    errors.append(
                        f"{f.name}: expected shape {f.shape}, got {actual_shape}"
                    )

            # Custom constraints
            for constraint in f.constraints:
                try:
                    if not constraint(value):
                        errors.append(f"{f.name}: constraint violated")
                except Exception as e:
                    errors.append(f"{f.name}: constraint error: {e}")

        return len(errors) == 0, errors

    def __repr__(self):
        fields_str = ", ".join(f.name for f in self.fields)
        return f"Schema({self.name}: {fields_str})"


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Dataset:
    """A typed dataset with schema validation."""
    name: str
    data: Any = None
    labels: Any = None
    schema: Optional[Schema] = None
    n_samples: int = 0

    def __repr__(self):
        return f"Dataset({self.name}, n={self.n_samples})"

    def __len__(self):
        return self.n_samples


def load_csv(path: str, schema: Optional[Schema] = None,
             delimiter: str = ",", has_header: bool = True) -> Dataset:
    """Load a CSV file into a Synaphe Dataset."""
    rows = []
    headers = None

    try:
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(delimiter)
                if i == 0 and has_header:
                    headers = [h.strip().strip('"') for h in parts]
                    continue
                row = []
                for val in parts:
                    val = val.strip().strip('"')
                    try:
                        row.append(float(val))
                    except ValueError:
                        row.append(val)
                rows.append(row)
    except FileNotFoundError:
        print(f"[synaphe] File not found: {path}")
        return Dataset(name=path, n_samples=0)

    if HAS_NUMPY:
        try:
            data = np.array(rows, dtype=np.float32)
        except (ValueError, TypeError):
            data = rows
    else:
        data = rows

    dataset = Dataset(
        name=os.path.basename(path),
        data=data,
        schema=schema,
        n_samples=len(rows)
    )

    # Validate against schema if provided
    if schema and headers:
        print(f"[synaphe] Validating {len(rows)} records against {schema.name}...")
        violations = 0
        for row in rows[:10]:  # Spot check first 10
            record = {h: v for h, v in zip(headers, row)}
            valid, errs = schema.validate(record)
            if not valid:
                violations += 1
        if violations:
            print(f"[synaphe] ⚠ {violations} schema violations in first 10 records")
        else:
            print(f"[synaphe] ✓ Schema validation passed")

    return dataset


def generate_synthetic(n_samples: int = 1000, n_features: int = 4,
                       n_classes: int = 2, noise: float = 0.1) -> Dataset:
    """Generate a synthetic classification dataset for testing."""
    if HAS_NUMPY:
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        weights = np.random.randn(n_features, n_classes)
        logits = data @ weights + noise * np.random.randn(n_samples, n_classes)
        labels = np.argmax(logits, axis=1)
    else:
        data = [[random.gauss(0, 1) for _ in range(n_features)]
                for _ in range(n_samples)]
        labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]

    return Dataset(
        name=f"synthetic_{n_samples}x{n_features}",
        data=data,
        labels=labels,
        n_samples=n_samples
    )


def load_mnist_sample(n: int = 100) -> Dataset:
    """Generate MNIST-like synthetic data for testing."""
    if HAS_NUMPY:
        data = np.random.rand(n, 28, 28).astype(np.float32)
        labels = np.random.randint(0, 10, size=n)
    else:
        data = [[[random.random() for _ in range(28)]
                 for _ in range(28)] for _ in range(n)]
        labels = [random.randint(0, 9) for _ in range(n)]

    return Dataset(
        name=f"mnist_sample_{n}",
        data=data,
        labels=labels,
        n_samples=n
    )


# ═══════════════════════════════════════════════════════════════════
# DATA TRANSFORMS (work in pipelines)
# ═══════════════════════════════════════════════════════════════════

def normalize(data, mean: float = 0.0, std: float = 1.0):
    """Normalize data to mean=0, std=1 (or specified values)."""
    if HAS_NUMPY and isinstance(data, np.ndarray):
        data_mean = np.mean(data) if mean == 0.0 else mean
        data_std = np.std(data) if std == 1.0 else std
        return (data - data_mean) / (data_std + 1e-8)
    if isinstance(data, list):
        flat = [x for row in data for x in (row if isinstance(row, list) else [row])]
        m = sum(flat) / len(flat)
        s = (sum((x - m) ** 2 for x in flat) / len(flat)) ** 0.5
        if isinstance(data[0], list):
            return [[(x - m) / (s + 1e-8) for x in row] for row in data]
        return [(x - m) / (s + 1e-8) for x in data]
    return data


def batch(data, size: int = 32):
    """Split data into batches."""
    if HAS_NUMPY and isinstance(data, np.ndarray):
        n = len(data)
        return [data[i:i+size] for i in range(0, n, size)]
    if isinstance(data, list):
        return [data[i:i+size] for i in range(0, len(data), size)]
    return [data]


def shuffle(data, seed: int = None):
    """Randomly shuffle data."""
    if seed is not None:
        random.seed(seed)

    if HAS_NUMPY and isinstance(data, np.ndarray):
        idx = np.random.permutation(len(data))
        return data[idx]
    if isinstance(data, list):
        result = list(data)
        random.shuffle(result)
        return result
    return data


def flatten(data):
    """Flatten tensor dimensions (keep batch dim)."""
    if HAS_NUMPY and isinstance(data, np.ndarray):
        if data.ndim > 2:
            return data.reshape(data.shape[0], -1)
        return data
    return data


def one_hot(labels, n_classes: int = 10):
    """Convert integer labels to one-hot vectors."""
    if HAS_NUMPY:
        labels = np.array(labels)
        result = np.zeros((len(labels), n_classes), dtype=np.float32)
        result[np.arange(len(labels)), labels] = 1.0
        return result
    return [[1.0 if j == label else 0.0 for j in range(n_classes)]
            for label in labels]


def validate(data, schema: Schema):
    """Validate data against a schema. Raises on failure."""
    if isinstance(data, dict):
        valid, errors = schema.validate(data)
        if not valid:
            raise ValueError(f"Schema validation failed: {errors}")
    return data


def split(dataset: Dataset, ratio: float = 0.8, seed: int = 42
          ) -> Tuple[Dataset, Dataset]:
    """Split a dataset into train and test sets."""
    random.seed(seed)
    n = dataset.n_samples
    split_idx = int(n * ratio)

    if HAS_NUMPY and isinstance(dataset.data, np.ndarray):
        idx = np.random.permutation(n)
        train_data = dataset.data[idx[:split_idx]]
        test_data = dataset.data[idx[split_idx:]]
        train_labels = dataset.labels[idx[:split_idx]] if dataset.labels is not None else None
        test_labels = dataset.labels[idx[split_idx:]] if dataset.labels is not None else None
    else:
        indices = list(range(n))
        random.shuffle(indices)
        train_data = [dataset.data[i] for i in indices[:split_idx]]
        test_data = [dataset.data[i] for i in indices[split_idx:]]
        train_labels = ([dataset.labels[i] for i in indices[:split_idx]]
                        if dataset.labels else None)
        test_labels = ([dataset.labels[i] for i in indices[split_idx:]]
                       if dataset.labels else None)

    train = Dataset(
        name=f"{dataset.name}_train", data=train_data,
        labels=train_labels, n_samples=split_idx
    )
    test = Dataset(
        name=f"{dataset.name}_test", data=test_data,
        labels=test_labels, n_samples=n - split_idx
    )
    return train, test


# ═══════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════

__all__ = [
    'Schema', 'SchemaField', 'Dataset',
    'load_csv', 'generate_synthetic', 'load_mnist_sample',
    'normalize', 'batch', 'shuffle', 'flatten', 'one_hot',
    'validate', 'split',
]
