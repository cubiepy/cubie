"""Cache parsed CellML models on disk."""

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union
import pickle
from hashlib import sha256
import json
import time

from cubie.cache_root import get_cache_root
from cubie.time_logger import default_timelogger


class CellMLCache:
    """Cache up to five parsed configurations per CellML model."""

    def __init__(self, model_name: str, cellml_path: str) -> None:
        """Initialize cache manager for a CellML model.

        Parameters
        ----------
        model_name : str
            Name used for cache directory (typically cellml filename stem)
        cellml_path : str
            Path to source CellML file

        Raises
        ------
        TypeError
            If model_name or cellml_path are not strings
        ValueError
            If model_name is empty string
        FileNotFoundError
            If cellml_path does not exist
        """
        # Validate model_name type
        if not isinstance(model_name, str):
            raise TypeError(
                f"model_name must be str, got {type(model_name).__name__}"
            )

        # Validate cellml_path type
        if not isinstance(cellml_path, str):
            raise TypeError(
                f"cellml_path must be str, got {type(cellml_path).__name__}"
            )

        # Validate model_name is not empty
        if not model_name:
            raise ValueError("model_name cannot be empty string")

        # Validate cellml_path exists
        cellml_path_obj = Path(cellml_path)
        if not cellml_path_obj.exists():
            raise FileNotFoundError(f"CellML file not found: {cellml_path}")

        self.model_name = model_name
        self.cellml_path = cellml_path
        self.cache_dir = get_cache_root() / model_name
        self.manifest_file = self.cache_dir / "cellml_cache_manifest.json"
        self.max_entries = 5  # LRU cache limit

    def get_cellml_hash(self) -> str:
        """Compute SHA256 hash of CellML file content.

        Reads entire file content and computes hash for cache validation.
        Whitespace changes will change the hash.

        Returns
        -------
        str
            Hexadecimal hash string (64 characters)

        Raises
        ------
        FileNotFoundError
            If CellML file has been deleted since initialization
        IOError
            If file cannot be read
        """
        # Read file content in binary mode for hashing
        with open(self.cellml_path, "rb") as f:
            content = f.read()

        # Compute SHA256 hash
        hash_obj = sha256(content)
        return hash_obj.hexdigest()

    def _serialize_args(
        self,
        parameters: Optional[
            Union[Iterable[str], Mapping[str, Any]]
        ],
        observables: Optional[Iterable[str]],
        precision,
        name: str,
        fix_singularities: bool = True,
        voltage_variable: Optional[str] = None,
        constant_values: Optional[Mapping[str, Any]] = None,
        parameter_values: Optional[Mapping[str, Any]] = None,
        initial_values: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Return a stable JSON cache-key payload."""
        # Handle precision consistently - convert numpy dtype to string name
        if precision is not None:
            precision_str = (
                precision.__name__
                if hasattr(precision, "__name__")
                else str(precision)
            )
        else:
            precision_str = "None"

        if isinstance(parameters, Mapping):
            serialized_parameters = self._serialize_values(parameters)
        else:
            serialized_parameters = (
                sorted(str(value) for value in parameters)
                if parameters
                else None
            )

        args_dict = {
            "parameters": serialized_parameters,
            "observables": sorted(observables) if observables else None,
            "precision": precision_str,
            "name": name,
            "fix_singularities": fix_singularities,
            "voltage_variable": voltage_variable,
            "constant_values": self._serialize_values(constant_values),
            "parameter_values": self._serialize_values(parameter_values),
            "initial_values": self._serialize_values(initial_values),
            "parse_format": 4,
        }
        return json.dumps(args_dict, sort_keys=True)

    @staticmethod
    def _serialize_values(
        values: Optional[Mapping[str, Any]],
    ) -> Optional[Dict[str, float]]:
        """Return sorted numeric values for the cache key."""

        if not values:
            return None
        return {
            str(key): float(values[key])
            for key in sorted(values, key=str)
        }

    def compute_cache_key(
        self,
        parameters: Optional[
            Union[Iterable[str], Mapping[str, Any]]
        ],
        observables: Optional[Iterable[str]],
        precision,
        name: str,
        fix_singularities: bool = True,
        voltage_variable: Optional[str] = None,
        constant_values: Optional[Mapping[str, Any]] = None,
        parameter_values: Optional[Mapping[str, Any]] = None,
        initial_values: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Compute cache key from file content hash and argument hash.

        Returns a short hash (first 16 chars) for use in filename.
        """
        file_hash = self.get_cellml_hash()
        args_str = self._serialize_args(
            parameters, observables, precision, name,
            fix_singularities=fix_singularities,
            voltage_variable=voltage_variable,
            constant_values=constant_values,
            parameter_values=parameter_values,
            initial_values=initial_values,
        )
        combined = file_hash + args_str
        return sha256(combined.encode()).hexdigest()[:16]

    def _load_manifest(self) -> dict:
        """Load manifest from disk or return empty structure."""
        if not self.manifest_file.exists():
            return {"version": 1, "file_hash": None, "entries": []}
        try:
            with open(self.manifest_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"version": 1, "file_hash": None, "entries": []}

    def _save_manifest(self, manifest: dict) -> None:
        """Save manifest to disk. Creates directory if needed.

        Caching is opportunistic - failures are logged but don't raise.
        """
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_file, "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            default_timelogger.print_message(
                f"Cache manifest save failed: {e}"
            )

    def _update_lru_order(self, manifest: dict, args_hash: str) -> dict:
        """Move args_hash to end of entries list (most recently used)."""
        entries = manifest.get("entries", [])
        # Remove existing entry if present
        entries = [e for e in entries if e.get("args_hash") != args_hash]
        # Add at end with updated timestamp
        new_entry = {"args_hash": args_hash, "last_used": time.time()}
        entries.append(new_entry)
        manifest["entries"] = entries
        return manifest

    def _evict_lru(self, manifest: dict) -> dict:
        """Remove oldest entry if over max_entries limit."""
        entries = manifest.get("entries", [])
        while len(entries) > self.max_entries:
            oldest = entries.pop(0)  # First entry is oldest
            # Delete the cache file
            cache_file = self.cache_dir / f"cache_{oldest['args_hash']}.pkl"
            try:
                cache_file.unlink()
            except FileNotFoundError:
                pass
        manifest["entries"] = entries
        return manifest

    def cache_valid(self, args_hash: str) -> bool:
        """Check if cache for given args_hash exists and file hash matches.

        Returns
        -------
        bool
            True if cache exists and is current, False otherwise
        """
        manifest = self._load_manifest()
        current_file_hash = self.get_cellml_hash()

        # If file hash changed, all caches are invalid
        if manifest.get("file_hash") != current_file_hash:
            return False

        # Check if args_hash is in entries
        entries = manifest.get("entries", [])
        for entry in entries:
            if entry.get("args_hash") == args_hash:
                cache_file = self.cache_dir / f"cache_{args_hash}.pkl"
                return cache_file.exists()
        return False

    def load_from_cache(self, args_hash: str) -> Optional[dict]:
        """Load cached data for given args_hash. Updates LRU order.

        Returns
        -------
        dict or None
            Dictionary with cached data. Returns None if load fails.
        """
        if not self.cache_valid(args_hash):
            return None

        cache_file = self.cache_dir / f"cache_{args_hash}.pkl"
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)

            # Update LRU order
            manifest = self._load_manifest()
            manifest = self._update_lru_order(manifest, args_hash)
            self._save_manifest(manifest)

            return cached_data
        except Exception as e:
            default_timelogger.print_message(f"Cache load error: {e}")
            return None

    def save_to_cache(
        self,
        args_hash: str,
        parsed_equations,
        indexed_bases,
        all_symbols: dict,
        user_functions: Optional[dict],
        fn_hash: str,
        precision,
        name: str,
        mass=None,
        definition=None,
    ) -> None:
        """Save cached data for given args_hash. Handles LRU eviction.

        Parameters
        ----------
        args_hash : str
            Cache key computed from arguments
        parsed_equations : ParsedEquations
            Equation container from parse_input
        indexed_bases : IndexedBases
            Index maps from parse_input
        all_symbols : dict
            Symbol mapping from parse_input
        user_functions : dict or None
            User-provided functions (may be None)
        fn_hash : str
            System hash from parse_input
        precision : PrecisionDType
            Floating-point precision
        name : str
            Model name
        mass : ndarray or None
            Solver mass matrix from structural simplification;
            ``None`` implies identity.
        definition : NormalisedSystemDefinition or None
            Constants-symbolic checkpoint from parse_input, used to
            re-specialise the system on constant changes.
        """
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Build cache dictionary
            cache_data = {
                "parsed_equations": parsed_equations,
                "indexed_bases": indexed_bases,
                "all_symbols": all_symbols,
                "user_functions": user_functions,
                "fn_hash": fn_hash,
                "precision": precision,
                "name": name,
                "mass": mass,
                "definition": definition,
            }

            # Save pickle file
            cache_file = self.cache_dir / f"cache_{args_hash}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Update manifest
            manifest = self._load_manifest()
            manifest["file_hash"] = self.get_cellml_hash()
            manifest = self._update_lru_order(manifest, args_hash)
            manifest = self._evict_lru(manifest)
            self._save_manifest(manifest)

        except Exception as e:
            default_timelogger.print_message(f"Cache save failed: {e}")
