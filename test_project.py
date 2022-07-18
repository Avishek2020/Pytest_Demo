import pandas as pd
import pytest

from prediction.lstm.project import add, add_col, subtract


def test_add():
    # test basic functionality
    assert add(1, 1) == 2
    # try a border case
    assert add(-1, -1) == -2


# test parameterization
@pytest.mark.parametrize("a,b,expected", [[2, 1, 1], [-1, 1, -2], [-2, 2, -4]])
def test_subtract(a, b, expected):
    assert subtract(a, b) == expected


def test_add_col():
    # start
    df = pd.DataFrame(
        [[0, 1], [2, 3]],
        index=['cat', 'dog'],
        columns=['weight', 'height'])
    # expected
    df_expected = pd.DataFrame(
        [[0, 1, 2], [2, 3, 2]],
        index=['cat', 'dog'],
        columns=['weight', 'height', 'n_ears'])
    # test the result
    assert add_col(df, 'n_ears', 2).equals(df_expected)


@pytest.fixture()
def df_avi():
    df = pd.DataFrame([[00, 11], [2, 3]],
                      columns=['weight', 'height'])
    return df


def test_add_col2(df_avi):
    # expected
    df_expected = pd.DataFrame(
        [[0, 1, 3], [2, 3, 5]],
        columns=['weight', 'height', 'hop_height'])
    print(add_col(df_avi,'hop_height', [6, 5]))
    assert add_col(df_avi,
                   'hop_height',
                   [3, 5]).equals(df_expected)
