"""
Main module.

This module contains code that create and prints a polars DataFrame.
"""

import polars as pl


def hello_polars(name: str, age: int) -> pl.DataFrame:
    """
    Create and returns a DataFrame.

    Returns
    -------
    pl.DataFrame:
        A DataFrame containing name and age.

    """
    first_name = name.split(" ")[0]

    data = [
        {
            "name": first_name,
            "age": age,
        },
    ]

    df = pl.DataFrame(data)

    return df


if __name__ == "__main__":
    print(hello_polars("Kasper", 21))
