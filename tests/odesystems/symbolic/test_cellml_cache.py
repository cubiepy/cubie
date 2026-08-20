"""Unit tests for CellML caching functionality."""


import pytest
from pathlib import Path
from numpy import float64

from cubie.odesystems.symbolic.parsing.cellml_cache import CellMLCache


@pytest.fixture
def basic_cellml_path(cellml_fixtures_dir):
    """Return path to basic_ode.cellml fixture."""
    return str(cellml_fixtures_dir / "basic_ode.cellml")


@pytest.fixture
def tmp_cellml_file(tmp_path):
    """Create temporary CellML file for testing."""
    cellml_content = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/1.0#">
    <component name="test">
        <variable name="x" initial_value="1.0"/>
    </component>
</model>
"""
    tmp_file = tmp_path / "test.cellml"
    tmp_file.write_text(cellml_content)
    return str(tmp_file)


def test_cache_initialization_valid_inputs(basic_cellml_path):
    """Verify CellMLCache initializes with valid model_name and cellml_path.

    Tests that the cache object is created successfully with proper
    attributes when given valid string inputs for both parameters.
    """
    cache = CellMLCache(model_name="basic_ode", cellml_path=basic_cellml_path)

    # Verify attributes are set correctly
    assert cache.model_name == "basic_ode"
    assert cache.cellml_path == basic_cellml_path
    assert "basic_ode" in str(cache.cache_dir)


def test_cache_initialization_invalid_inputs(basic_cellml_path):
    """Verify TypeError raised for non-string inputs, ValueError for empty
    model_name.

    Tests input validation for the __init__ method, ensuring proper
    exceptions are raised for invalid input types and values.
    """
    # Test non-string model_name
    with pytest.raises(TypeError, match="model_name must be str"):
        CellMLCache(model_name=123, cellml_path=basic_cellml_path)

    # Test non-string cellml_path
    with pytest.raises(TypeError, match="cellml_path must be str"):
        CellMLCache(model_name="test", cellml_path=Path(basic_cellml_path))

    # Test empty model_name
    with pytest.raises(ValueError, match="model_name cannot be empty"):
        CellMLCache(model_name="", cellml_path=basic_cellml_path)

    # Test non-existent cellml_path
    with pytest.raises(FileNotFoundError, match="CellML file not found"):
        CellMLCache(model_name="test", cellml_path="/nonexistent/file.cellml")


def test_get_cellml_hash_consistent(tmp_cellml_file):
    """Verify hash is consistent for same file, changes when file changes.

    Tests that the SHA256 hash computation is deterministic for unchanged
    files and properly detects file modifications.
    """
    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)

    # Get hash twice - should be identical
    hash1 = cache.get_cellml_hash()
    hash2 = cache.get_cellml_hash()
    assert hash1 == hash2

    # Verify hash is 64 characters (SHA256 in hex)
    assert len(hash1) == 64

    # Modify file and verify hash changes
    with open(tmp_cellml_file, 'a') as f:
        f.write("\n<!-- Modified -->\n")

    hash3 = cache.get_cellml_hash()
    assert hash3 != hash1


def test_serialize_args_consistent(tmp_cellml_file):
    """Verify _serialize_args produces consistent output for same inputs.

    Tests that argument serialization is deterministic and produces
    the same JSON string for identical inputs.
    """
    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)

    # Test with None parameters and observables
    args1 = cache._serialize_args(None, None, float64, "test")
    args2 = cache._serialize_args(None, None, float64, "test")
    assert args1 == args2

    # Test with lists - order should be normalized
    params1 = ["param1", "param2", "param3"]
    params2 = ["param3", "param1", "param2"]  # Different order
    args3 = cache._serialize_args(params1, None, float64, "test")
    args4 = cache._serialize_args(params2, None, float64, "test")
    assert args3 == args4  # Should be same after sorting

    # Test with observables
    obs = ["obs1", "obs2"]
    args5 = cache._serialize_args(None, obs, float64, "test")
    assert "obs1" in args5
    assert "obs2" in args5


def test_compute_cache_key_different_args(tmp_cellml_file):
    """Verify compute_cache_key produces different keys for different args.

    Tests that different parameter configurations produce different
    cache keys, enabling multi-config caching.
    """
    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)

    # Different parameters should give different keys
    key1 = cache.compute_cache_key(None, None, float64, "test")
    key2 = cache.compute_cache_key(["param1"], None, float64, "test")
    assert key1 != key2

    # Different observables should give different keys
    key3 = cache.compute_cache_key(None, ["obs1"], float64, "test")
    assert key1 != key3
    assert key2 != key3

    # Same args should give same key
    key4 = cache.compute_cache_key(None, None, float64, "test")
    assert key1 == key4

    # Verify key is 16 characters (truncated hash)
    assert len(key1) == 16


def test_compute_cache_key_includes_edited_values(tmp_cellml_file):
    """Edited model values change the cache key."""

    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)
    base = cache.compute_cache_key(
        ["p"], None, float64, "test",
        constant_values={"c": 1.0},
        parameter_values={"p": 2.0},
        initial_values={"x": 3.0},
    )
    changed_constant = cache.compute_cache_key(
        ["p"], None, float64, "test",
        constant_values={"c": 4.0},
        parameter_values={"p": 2.0},
        initial_values={"x": 3.0},
    )
    changed_parameter = cache.compute_cache_key(
        ["p"], None, float64, "test",
        constant_values={"c": 1.0},
        parameter_values={"p": 4.0},
        initial_values={"x": 3.0},
    )
    changed_initial = cache.compute_cache_key(
        ["p"], None, float64, "test",
        constant_values={"c": 1.0},
        parameter_values={"p": 2.0},
        initial_values={"x": 4.0},
    )

    assert len(
        {base, changed_constant, changed_parameter, changed_initial}
    ) == 4


def test_compute_cache_key_includes_parameter_dict_values(
    tmp_cellml_file,
):
    """Parameter dictionary values change the request cache key."""

    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)
    first = cache.compute_cache_key(
        {"p": 1.0}, None, float64, "test"
    )
    second = cache.compute_cache_key(
        {"p": 2.0}, None, float64, "test"
    )
    assert first != second


def test_cache_valid_missing_file(basic_cellml_path, isolated_cache_root):
    """Verify cache_valid() returns False when cache file doesn't exist.

    Tests that cache validation correctly identifies when no cache file
    is present in the expected location.
    """
    cache = CellMLCache(model_name="basic_ode", cellml_path=basic_cellml_path)
    args_hash = cache.compute_cache_key(None, None, float64, "basic_ode")

    # Cache file should not exist yet
    assert not cache.manifest_file.exists()
    assert cache.cache_valid(args_hash) is False


def test_cache_valid_hash_mismatch(tmp_cellml_file, isolated_cache_root):
    """Verify cache_valid() returns False when file hash doesn't match.

    Tests cache invalidation when the source CellML file has been
    modified after cache creation.
    """
    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)
    args_hash = cache.compute_cache_key(None, None, float64, "test")

    # Create manifest with wrong file hash
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    wrong_manifest = {
        "version": 1,
        "file_hash": "wronghash123",
        "entries": [{"args_hash": args_hash, "last_used": 1234567890}]
    }
    cache._save_manifest(wrong_manifest)

    # Cache should be invalid due to file hash mismatch
    assert cache.cache_valid(args_hash) is False


def test_load_from_cache_returns_none_invalid(
    tmp_cellml_file, isolated_cache_root
):
    """Verify load_from_cache() returns None when cache invalid.

    Tests that attempting to load from an invalid or non-existent cache
    returns None rather than raising an exception.
    """
    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)
    args_hash = cache.compute_cache_key(None, None, float64, "test")

    # No cache file exists
    assert cache.load_from_cache(args_hash) is None

    # Create manifest with wrong file hash
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    wrong_manifest = {
        "version": 1,
        "file_hash": "wronghash",
        "entries": [{"args_hash": args_hash, "last_used": 1234567890}]
    }
    cache._save_manifest(wrong_manifest)

    # Should still return None
    assert cache.load_from_cache(args_hash) is None


def test_save_and_load_roundtrip(tmp_cellml_file, isolated_cache_root):
    """Verify data saved to cache can be loaded back correctly.

    Tests the complete save/load cycle, ensuring that all data saved to
    the cache can be successfully retrieved with the same values.
    """
    from sympy import symbols

    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)
    args_hash = cache.compute_cache_key(None, None, float64, "test")

    # Create minimal mock objects for testing
    # (In real use, these come from parse_input)
    x = symbols('x')

    # Create mock ParsedEquations with minimal required attributes
    # Using a simple dict to represent it for testing
    mock_equations = {"states": [x], "derivatives": [x]}

    # Create mock IndexedBases (simple dict representation)
    mock_indices = {"states": {"x": 0}}

    # Create test data
    all_symbols = {"x": x}
    user_functions = None
    fn_hash = "test_hash_12345"
    precision = float64
    name = "test"

    # Save to cache
    cache.save_to_cache(
        args_hash=args_hash,
        parsed_equations=mock_equations,
        indexed_bases=mock_indices,
        all_symbols=all_symbols,
        user_functions=user_functions,
        fn_hash=fn_hash,
        precision=precision,
        name=name,
    )

    # Verify cache file was created
    cache_file = cache.cache_dir / f"cache_{args_hash}.pkl"
    assert cache_file.exists()

    # Verify manifest was created
    assert cache.manifest_file.exists()

    # Verify cache is valid
    assert cache.cache_valid(args_hash) is True

    # Load from cache
    loaded_data = cache.load_from_cache(args_hash)

    # Verify data was loaded
    assert loaded_data is not None

    # Verify expected keys are present (note: no 'cellml_hash' in new format)
    assert 'parsed_equations' in loaded_data
    assert 'indexed_bases' in loaded_data
    assert 'all_symbols' in loaded_data
    assert 'user_functions' in loaded_data
    assert 'fn_hash' in loaded_data
    assert 'precision' in loaded_data
    assert 'name' in loaded_data

    # Verify data matches what was saved
    assert loaded_data['fn_hash'] == fn_hash
    assert loaded_data['name'] == name
    assert loaded_data['precision'] == precision
    assert loaded_data['user_functions'] is None


def test_corrupted_cache_returns_none(tmp_cellml_file, isolated_cache_root):
    """Verify load_from_cache() handles corrupted pickle gracefully.

    Tests that corrupted cache files are handled gracefully by returning
    None rather than raising exceptions that would crash the parsing.
    """
    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)
    args_hash = cache.compute_cache_key(None, None, float64, "test")

    # Create cache directory
    cache.cache_dir.mkdir(parents=True, exist_ok=True)

    # Get actual file hash
    actual_hash = cache.get_cellml_hash()

    # Create valid manifest
    manifest = {
        "version": 1,
        "file_hash": actual_hash,
        "entries": [{"args_hash": args_hash, "last_used": 1234567890}]
    }
    cache._save_manifest(manifest)

    # Write corrupted cache file (invalid pickle)
    cache_file = cache.cache_dir / f"cache_{args_hash}.pkl"
    with open(cache_file, 'wb') as f:
        f.write(b"corrupted pickle data that cannot be unpickled")

    # Cache should be valid (hash matches) but load should return None
    assert cache.cache_valid(args_hash) is True
    assert cache.load_from_cache(args_hash) is None


def test_lru_eviction_on_sixth_entry(tmp_cellml_file, isolated_cache_root):
    """Verify LRU eviction when 6th config added to cache.

    Tests that the cache correctly evicts the oldest entry when
    the max_entries limit (5) is exceeded.
    """
    from sympy import symbols

    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)

    # Create 6 different configurations
    configs = [
        (None, None, float64, "test"),
        (["param1"], None, float64, "test"),
        (None, ["obs1"], float64, "test"),
        (["param1"], ["obs1"], float64, "test"),
        (["param1", "param2"], None, float64, "test"),
        (None, ["obs1", "obs2"], float64, "test"),
    ]

    # Create minimal test data
    x = symbols('x')
    mock_equations = {"states": [x]}
    mock_indices = {"states": {"x": 0}}
    all_symbols = {"x": x}

    # Save all 6 configs
    hashes = []
    for params, obs, prec, name in configs:
        args_hash = cache.compute_cache_key(params, obs, prec, name)
        hashes.append(args_hash)
        cache.save_to_cache(
            args_hash=args_hash,
            parsed_equations=mock_equations,
            indexed_bases=mock_indices,
            all_symbols=all_symbols,
            user_functions=None,
            fn_hash="test_hash",
            precision=prec,
            name=name,
        )

    # Load manifest and check entries
    manifest = cache._load_manifest()
    entries = manifest.get("entries", [])

    # Should have max_entries (5) entries
    assert len(entries) == 5

    # First config (oldest) should have been evicted
    entry_hashes = [e["args_hash"] for e in entries]
    assert hashes[0] not in entry_hashes

    # Other 5 configs should still be present
    for h in hashes[1:]:
        assert h in entry_hashes

    # First config's cache file should be deleted
    first_cache_file = cache.cache_dir / f"cache_{hashes[0]}.pkl"
    assert not first_cache_file.exists()


def test_cache_hit_with_different_params(tmp_cellml_file, isolated_cache_root):
    """Verify cache can store and retrieve different parameter sets.

    Tests that the cache correctly handles multiple configurations
    for the same CellML file.
    """
    from sympy import symbols

    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)

    # Create minimal test data
    x = symbols('x')
    mock_equations = {"states": [x]}
    mock_indices = {"states": {"x": 0}}
    all_symbols = {"x": x}

    # Save config 1: no params/obs
    hash1 = cache.compute_cache_key(None, None, float64, "test")
    cache.save_to_cache(
        args_hash=hash1,
        parsed_equations=mock_equations,
        indexed_bases=mock_indices,
        all_symbols=all_symbols,
        user_functions=None,
        fn_hash="hash1",
        precision=float64,
        name="test",
    )

    # Save config 2: with params
    hash2 = cache.compute_cache_key(["param1"], None, float64, "test")
    cache.save_to_cache(
        args_hash=hash2,
        parsed_equations=mock_equations,
        indexed_bases=mock_indices,
        all_symbols=all_symbols,
        user_functions=None,
        fn_hash="hash2",
        precision=float64,
        name="test",
    )

    # Both should be valid
    assert cache.cache_valid(hash1) is True
    assert cache.cache_valid(hash2) is True

    # Load both and verify correct data
    data1 = cache.load_from_cache(hash1)
    data2 = cache.load_from_cache(hash2)

    assert data1 is not None
    assert data2 is not None
    assert data1['fn_hash'] == "hash1"
    assert data2['fn_hash'] == "hash2"


def test_file_hash_change_invalidates_all_configs(
    tmp_cellml_file, isolated_cache_root
):
    """Verify that file hash change invalidates all cached configs.

    Tests that when the source CellML file is modified, all cached
    configurations become invalid regardless of their args_hash.
    """
    from sympy import symbols

    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)

    # Create minimal test data
    x = symbols('x')
    mock_equations = {"states": [x]}
    mock_indices = {"states": {"x": 0}}
    all_symbols = {"x": x}

    # Save two different configs
    hash1 = cache.compute_cache_key(None, None, float64, "test")
    hash2 = cache.compute_cache_key(["param1"], None, float64, "test")

    for h in [hash1, hash2]:
        cache.save_to_cache(
            args_hash=h,
            parsed_equations=mock_equations,
            indexed_bases=mock_indices,
            all_symbols=all_symbols,
            user_functions=None,
            fn_hash="test_hash",
            precision=float64,
            name="test",
        )

    # Both should be valid
    assert cache.cache_valid(hash1) is True
    assert cache.cache_valid(hash2) is True

    # Modify the CellML file
    with open(tmp_cellml_file, 'a') as f:
        f.write("\n<!-- Modified -->\n")

    # Both configs should now be invalid
    assert cache.cache_valid(hash1) is False
    assert cache.cache_valid(hash2) is False


@pytest.fixture
def isolated_cache(basic_cellml_path, tmp_path):
    """Return a CellMLCache whose storage is redirected to tmp_path."""
    cache = CellMLCache(
        model_name="basic_ode", cellml_path=basic_cellml_path
    )
    cache.cache_dir = tmp_path / "cache"
    cache.manifest_file = cache.cache_dir / "manifest.json"
    return cache


def test_compute_cache_key_with_none_precision(isolated_cache):
    """A None precision serialises without error."""
    key = isolated_cache.compute_cache_key(None, None, None, "model")
    assert isinstance(key, str)
    assert len(key) == 16


def test_load_manifest_recovers_from_corrupt_json(isolated_cache):
    """A corrupt manifest file yields the default empty structure."""
    isolated_cache.cache_dir.mkdir(parents=True, exist_ok=True)
    isolated_cache.manifest_file.write_text("{not valid json")
    manifest = isolated_cache._load_manifest()
    assert manifest == {"version": 1, "file_hash": None, "entries": []}


def test_save_manifest_swallows_directory_conflict(
    isolated_cache, tmp_path
):
    """_save_manifest logs and returns when the directory cannot be made."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file")
    # Point cache_dir at an existing file so mkdir raises.
    isolated_cache.cache_dir = blocker
    isolated_cache.manifest_file = blocker / "manifest.json"
    # Should not raise despite the impossible directory.
    isolated_cache._save_manifest({"version": 1, "entries": []})


def test_evict_lru_ignores_missing_cache_file(isolated_cache):
    """Eviction tolerates a cache file that is already gone."""
    isolated_cache.max_entries = 1
    manifest = {
        "version": 1,
        "file_hash": None,
        "entries": [
            {"args_hash": "gone_a", "last_used": 1.0},
            {"args_hash": "gone_b", "last_used": 2.0},
        ],
    }
    result = isolated_cache._evict_lru(manifest)
    assert len(result["entries"]) == 1
    assert result["entries"][0]["args_hash"] == "gone_b"


def test_save_to_cache_swallows_pickle_failure(isolated_cache):
    """save_to_cache logs and returns when the payload is unpicklable."""
    unpicklable = {"fn": lambda value: value}
    # A lambda in the payload makes pickle.dump raise; the method must
    # swallow it rather than propagate.
    isolated_cache.save_to_cache(
        "hash123",
        parsed_equations=None,
        indexed_bases=None,
        all_symbols=unpicklable,
        user_functions=unpicklable,
        fn_hash="abc",
        precision=float64,
        name="model",
    )
