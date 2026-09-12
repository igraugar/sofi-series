"""Loading of univariate time series classification datasets.

Three sources are tried in turn and any of them alone is enough, so the package
works offline as readily as it works with a network connection.

1. A local directory holding ``X_train.npy``, ``y_train.npy``, ``X_valid.npy``
   and ``y_valid.npy``, which is the layout the bundled sample follows. This
   path needs nothing beyond ``numpy`` and it is the one to prefer whenever the
   data must travel with the code.
2. The ``tsai`` loader, when that package is installed. It is never required,
   and importing it is attempted only when a dataset is requested by name.
3. A direct download of the archive of the dataset, cached on disk so that the
   network is consulted once.

The loader returns a small record rather than a tuple, since four arrays and a
class list are easier to carry together than to keep in order.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "TimeSeriesDataset",
    "load_dataset",
    "load_local",
    "load_ts",
    "bundled_path",
]

_ARCHIVE_URL = "http://www.timeseriesclassification.com/aeon-toolkit/{name}.zip"


@dataclass
class TimeSeriesDataset:
    """Train and test split of one classification problem.

    Attributes
    ----------
    X_train, X_test : ndarray
        Instances shaped ``(n, n_channels, n_timestamps)``.
    y_train, y_test : ndarray
        Labels, kept in their original type so that reports stay readable.
    classes : list
        Distinct labels in ascending order, which is the order of the
        probability columns of a classifier trained on ``y_train``.
    name : str
        Name of the problem, used in the summary line.
    source : str
        Where the arrays came from, one of ``"local"``, ``"ts"``, ``"tsai"`` or
        ``"archive"``.
    """

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    classes: List = field(default_factory=list)
    source: str = "local"

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_channels(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def n_timestamps(self) -> int:
        return int(self.X_train.shape[2])

    def codes(self, y) -> np.ndarray:
        """Positions of labels among the classes, as a classifier orders them.

        Parameters
        ----------
        y : array
            Labels drawn from the classes of the dataset.

        Returns
        -------
        ndarray
            Their column positions, which is what a classifier predicts.
        """
        lookup = {str(value): position for position, value in enumerate(self.classes)}
        return np.array([lookup[str(value)] for value in np.asarray(y)], dtype=int)

    def summary(self) -> str:
        return (
            f"{self.name}, {self.n_classes} classes, "
            f"{len(self.X_train)} train and {len(self.X_test)} test instances, "
            f"{self.n_channels} channel(s) of {self.n_timestamps} points "
            f"[{self.source}]"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TimeSeriesDataset({self.summary()})"


def _shape(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if array.ndim == 2:
        return array[:, None, :]
    if array.ndim == 3:
        return array
    raise ValueError(f"Instances must be two or three dimensional, got {array.shape}.")


def bundled_path() -> Optional[Path]:
    """Directory of the sample data shipped with the package, when present."""
    candidate = Path(__file__).resolve().parent / "datasets"
    return candidate if candidate.is_dir() else None


def load_local(directory, name: Optional[str] = None) -> TimeSeriesDataset:
    """Load a dataset from four ``.npy`` files in one directory.

    The expected names are ``X_train``, ``y_train``, ``X_valid`` and
    ``y_valid``, with ``X_test`` and ``y_test`` accepted as synonyms of the last
    two. Instances may be stored as ``(n, n_timestamps)`` or as ``(n,
    n_channels, n_timestamps)``.

    Parameters
    ----------
    directory : path
        Folder holding the four files.
    name : str, optional
        Name given to the dataset. The name of the folder is used by default.

    Returns
    -------
    TimeSeriesDataset
        The four arrays and the class list.
    """
    folder = Path(directory)
    if not folder.is_dir():
        raise FileNotFoundError(f"No directory was found at {folder}.")

    def read(*candidates):
        for candidate in candidates:
            path = folder / f"{candidate}.npy"
            if path.exists():
                return np.load(path, allow_pickle=True)
        raise FileNotFoundError(
            f"None of {candidates} was found in {folder}. The loader expects "
            "X_train, y_train and either X_valid or X_test, with their labels."
        )

    X_train, y_train = read("X_train"), read("y_train")
    X_test, y_test = read("X_valid", "X_test"), read("y_valid", "y_test")
    classes = sorted({str(v) for v in np.concatenate([y_train, y_test])})
    return TimeSeriesDataset(
        name=name or folder.name,
        X_train=_shape(X_train),
        y_train=np.asarray(y_train),
        X_test=_shape(X_test),
        y_test=np.asarray(y_test),
        classes=classes,
        source="local",
    )


def _parse_ts(path) -> Tuple[np.ndarray, np.ndarray]:
    """Read one file in the ``.ts`` format of the UCR and UEA archives.

    Each data row holds one channel per colon separated field, with the label
    last, so univariate and multivariate problems are read by the same code.
    """
    rows, in_data = [], False
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.lower().startswith("@data"):
                in_data = True
                continue
            if not in_data or stripped.startswith("@"):
                continue
            rows.append(stripped)
    if not rows:
        raise ValueError(f"The file {path} holds no data rows.")

    values, labels = [], []
    for row in rows:
        fields = row.split(":")
        labels.append(fields[-1].strip().strip("'\""))
        values.append([[float(v) for v in field.split(",")] for field in fields[:-1]])
    return np.array(values, dtype=float), np.array(labels)


def load_ts(train_path, test_path, name: Optional[str] = None) -> TimeSeriesDataset:
    """Load a problem from a pair of ``.ts`` files.

    The format is the one distributed by the UCR and UEA archives, and the
    reader handles univariate and multivariate problems alike.
    """
    X_train, y_train = _parse_ts(train_path)
    X_test, y_test = _parse_ts(test_path)
    classes = sorted({str(v) for v in np.concatenate([y_train, y_test])})
    return TimeSeriesDataset(
        name=name or Path(train_path).stem.replace("_TRAIN", ""),
        X_train=_shape(X_train),
        y_train=y_train,
        X_test=_shape(X_test),
        y_test=y_test,
        classes=classes,
        source="ts",
    )


def _load_with_tsai(name: str) -> TimeSeriesDataset:
    from tsai.data.external import get_UCR_data  # noqa: F401

    X_train, y_train, X_test, y_test = get_UCR_data(name, return_split=True)
    classes = sorted({str(v) for v in np.concatenate([y_train, y_test])})
    return TimeSeriesDataset(
        name=name,
        X_train=_shape(X_train),
        y_train=np.asarray(y_train),
        X_test=_shape(X_test),
        y_test=np.asarray(y_test),
        classes=classes,
        source="tsai",
    )


def _parse_arff(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal reader for the flat ARFF files of the univariate archive."""
    rows, in_data = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        lowered = stripped.lower()
        if lowered.startswith("@data"):
            in_data = True
            continue
        if not in_data or stripped.startswith("@"):
            continue
        rows.append(stripped.split(","))
    if not rows:
        raise ValueError("The archive holds no data rows in the expected format.")
    values = np.array([row[:-1] for row in rows], dtype=float)
    labels = np.array([row[-1].strip().strip("'\"") for row in rows])
    return values, labels


def _load_from_archive(name: str, cache: Path) -> TimeSeriesDataset:
    import urllib.request

    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{name}.zip"
    if not archive.exists():
        url = _ARCHIVE_URL.format(name=name)
        with urllib.request.urlopen(url, timeout=60) as response:
            archive.write_bytes(response.read())

    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()

        def pick(suffix):
            matches = [n for n in names if n.upper().endswith(suffix)]
            if not matches:
                raise FileNotFoundError(
                    f"The archive of {name} holds no file ending in {suffix}."
                )
            return matches[0]

        train_text = bundle.read(pick("_TRAIN.ARFF")).decode("utf-8", "ignore")
        test_text = bundle.read(pick("_TEST.ARFF")).decode("utf-8", "ignore")

    X_train, y_train = _parse_arff(train_text)
    X_test, y_test = _parse_arff(test_text)
    classes = sorted({str(v) for v in np.concatenate([y_train, y_test])})
    return TimeSeriesDataset(
        name=name,
        X_train=_shape(X_train),
        y_train=y_train,
        X_test=_shape(X_test),
        y_test=y_test,
        classes=classes,
        source="archive",
    )


def load_dataset(
    name: str,
    directory=None,
    cache=None,
    prefer: str = "local",
) -> TimeSeriesDataset:
    """Load a dataset by name, from disk or from the archive.

    Parameters
    ----------
    name : str
        Name of the dataset, such as ``"CBF"``.
    directory : path, optional
        Folder holding the ``.npy`` files. The folder bundled with the package
        is consulted when the argument is omitted, so the sample loads offline.
    cache : path, optional
        Where downloaded archives are kept. The default is
        ``~/.sofits/datasets``.
    prefer : {"local", "tsai", "archive"}, default="local"
        Source tried first. The remaining ones follow as fallbacks, and the
        error raised when all of them fail names every attempt.

    Returns
    -------
    TimeSeriesDataset
        The four arrays and the class list.
    """
    cache = Path(cache) if cache is not None else Path.home() / ".sofits" / "datasets"
    roots = []
    if directory is not None:
        roots.append(Path(directory))
    bundled = bundled_path()
    if bundled is not None:
        roots.append(bundled)
    roots.append(cache)

    def try_local():
        for root in roots:
            folder = root / name if (root / name).is_dir() else root
            if (folder / "X_train.npy").exists():
                return load_local(folder, name=name)
            train = folder / f"{name}_TRAIN.ts"
            test = folder / f"{name}_TEST.ts"
            if train.exists() and test.exists():
                return load_ts(train, test, name=name)
        raise FileNotFoundError(
            f"No local copy of {name} was found under {[str(r) for r in roots]}."
        )

    attempts = {"local": try_local, "tsai": lambda: _load_with_tsai(name),
                "archive": lambda: _load_from_archive(name, cache)}
    order = [prefer] + [key for key in ("local", "tsai", "archive") if key != prefer]

    failures = []
    for key in order:
        try:
            return attempts[key]()
        except Exception as error:  # pragma: no cover - depends on the machine
            failures.append(f"{key} gave {type(error).__name__}, {error}")
    raise FileNotFoundError(
        f"The dataset {name} could not be loaded. Attempts, " + " | ".join(failures)
    )
