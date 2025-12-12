import tempfile
from pathlib import Path

import pytest
import yaml

from bbml.core.utils.config_compose import config_compose, deep_update, merge_lists


class TestDeepUpdate:
    def test_simple_update(self):
        base = {"a": 1, "b": 2}
        updates = {"b": 3, "c": 4}
        result = deep_update(base, updates)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_dict_merge(self):
        base = {"outer": {"a": 1, "b": 2}}
        updates = {"outer": {"b": 3, "c": 4}}
        result = deep_update(base, updates)
        assert result == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_deeply_nested(self):
        base = {"l1": {"l2": {"l3": {"a": 1}}}}
        updates = {"l1": {"l2": {"l3": {"b": 2}}}}
        result = deep_update(base, updates)
        assert result == {"l1": {"l2": {"l3": {"a": 1, "b": 2}}}}

    def test_list_replace_strategy(self):
        base = {"items": [1, 2, 3]}
        updates = {"items": [4, 5]}
        result = deep_update(base, updates, list_strategy="replace")
        assert result == {"items": [4, 5]}

    def test_list_concat_strategy(self):
        base = {"items": [1, 2]}
        updates = {"items": [3, 4]}
        result = deep_update(base, updates, list_strategy="concat")
        assert result == {"items": [1, 2, 3, 4]}


class TestMergeLists:
    def test_replace(self):
        assert merge_lists([1, 2], [3, 4], "replace") == [3, 4]

    def test_concat(self):
        assert merge_lists([1, 2], [3, 4], "concat") == [1, 2, 3, 4]

    def test_elementwise_override(self):
        result = merge_lists([1, 2, 3], [10, 20], "elementwise")
        assert result == [10, 20, 3]

    def test_elementwise_extend(self):
        result = merge_lists([1], [10, 20, 30], "elementwise")
        assert result == [10, 20, 30]

    def test_elementwise_dict_merge(self):
        base = [{"a": 1}, {"b": 2}]
        updates = [{"a": 10, "c": 3}]
        result = merge_lists(base, updates, "elementwise")
        assert result == [{"a": 10, "c": 3}, {"b": 2}]

    def test_custom_callable(self):
        custom_fn = lambda base, update: base + update + [999]
        result = merge_lists([1], [2], custom_fn)
        assert result == [1, 2, 999]

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown list strategy"):
            merge_lists([1], [2], "invalid")


class TestConfigCompose:
    def test_compose_from_dicts(self):
        configs = [
            {"a": 1, "b": 2},
            {"b": 3, "c": 4},
        ]
        result = config_compose(configs)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_compose_from_yaml_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first yaml
            path1 = Path(tmpdir) / "config1.yaml"
            path1.write_text(yaml.dump({"a": 1, "nested": {"x": 10}}))

            # Create second yaml
            path2 = Path(tmpdir) / "config2.yaml"
            path2.write_text(yaml.dump({"b": 2, "nested": {"y": 20}}))

            result = config_compose([path1, path2])
            assert result == {"a": 1, "b": 2, "nested": {"x": 10, "y": 20}}

    def test_compose_mixed_dict_and_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "config.yaml"
            path1.write_text(yaml.dump({"a": 1}))

            result = config_compose([path1, {"b": 2}])
            assert result == {"a": 1, "b": 2}

    def test_compose_empty_list(self):
        result = config_compose([])
        assert result == {}

    def test_compose_with_list_strategy(self):
        configs = [
            {"items": [1, 2]},
            {"items": [3, 4]},
        ]
        result = config_compose(configs, list_strategy="concat")
        assert result == {"items": [1, 2, 3, 4]}

    def test_compose_string_iterates_chars(self):
        # Note: strings are Sequences in Python, so config_compose
        # will iterate over characters. This tests current behavior.
        # Each char 'a', 'b', 'c' is treated as a file path.
        # This will fail with FileNotFoundError, which is expected.
        with pytest.raises(FileNotFoundError):
            config_compose("abc")  # type: ignore

