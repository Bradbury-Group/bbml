import pytest

from bbml.core.registry import Registry


class TestRegistry:
    def test_init(self):
        reg = Registry[int]("Test")
        assert reg.name == "Test"
        assert len(reg) == 0

    def test_add_and_get(self):
        reg = Registry[int]("Test")
        reg.add("foo", 42)
        assert reg.get("foo") == 42
        assert reg.get("bar") is None
        assert reg.get("bar", default=99) == 99

    def test_add_duplicate_raises(self):
        reg = Registry[int]("Test")
        reg.add("foo", 1)
        with pytest.raises(KeyError, match="already registered"):
            reg.add("foo", 2)

    def test_add_force_overwrites(self):
        reg = Registry[int]("Test")
        reg.add("foo", 1)
        reg.add("foo", 2, force=True)
        assert reg["foo"] == 2

    def test_remove(self):
        reg = Registry[int]("Test")
        reg.add("foo", 1)
        reg.remove("foo")
        assert "foo" not in reg

    def test_contains(self):
        reg = Registry[int]("Test")
        reg.add("foo", 1)
        assert "foo" in reg
        assert "bar" not in reg

    def test_getitem(self):
        reg = Registry[int]("Test")
        reg.add("foo", 42)
        assert reg["foo"] == 42

    def test_len(self):
        reg = Registry[int]("Test")
        assert len(reg) == 0
        reg.add("a", 1)
        reg.add("b", 2)
        assert len(reg) == 2

    def test_iter(self):
        reg = Registry[int]("Test")
        reg.add("a", 1)
        reg.add("b", 2)
        assert set(reg) == {"a", "b"}

    def test_keys_values_items(self):
        reg = Registry[int]("Test")
        reg.add("a", 1)
        reg.add("b", 2)
        assert set(reg.keys()) == {"a", "b"}
        assert set(reg.values()) == {1, 2}
        assert set(reg.items()) == {("a", 1), ("b", 2)}

    def test_register_decorator_with_name(self):
        reg = Registry[type]("Test")

        @reg.register("MyClass")
        class Foo:
            pass

        assert "MyClass" in reg
        assert reg["MyClass"] is Foo

    def test_register_decorator_without_name(self):
        reg = Registry[type]("Test")

        @reg.register()
        class Bar:
            pass

        # Registry uses __qualname__ which includes full path
        registered_key = list(reg.keys())[0]
        assert "Bar" in registered_key
        assert reg[registered_key] is Bar

    def test_register_decorator_bare(self):
        reg = Registry[type]("Test")

        @reg.register
        class Baz:
            pass

        # Registry uses __qualname__ which includes full path
        registered_key = list(reg.keys())[0]
        assert "Baz" in registered_key
        assert reg[registered_key] is Baz

    def test_repr(self):
        reg = Registry[int]("Test")
        reg.add("foo", 1)
        assert "TestRegistry" in repr(reg)
        assert "foo" in repr(reg)

