"""
Main module tests.

This is the tests for the main module.
"""

from package.main import hello_polars


def test_hello_ploars():
    """Tests the hello_polars() function."""
    name = "Kasper"
    age = 21

    df = hello_polars(name, age)

    assert df["name"][0] == name
    assert df["age"][0] == age
