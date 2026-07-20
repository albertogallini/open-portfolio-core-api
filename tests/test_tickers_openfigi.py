import unittest

from oport.openbb import obb
from oport.const_and_utils import *
from pandas.testing import assert_frame_equal
from oport.fields import *
from oport.functions.functions_positions import get_tickers_and_suffix
import warnings


class TestOportTickersOpenFigi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Suppress DeprecationWarnings
        warnings.simplefilter("ignore", DeprecationWarning)

    def test_fetching_bbgid(self):
        data = pd.DataFrame(
            {
                "isin": [
                    "IE00BGBN6P67",
                    "IE00B4L5Y983",
                    "US0231351067",
                    "US46222L1089",
                    "NL0011585146",
                ],
                "currency": ["EUR", "EUR", "USD", "USD", "EUR"],
                "figi": [
                    "BBG00NHX4J20",
                    "BBG000PH98P0",
                    "BBG000BVPV84",
                    "BBG00XZP0LB4",
                    "BBG009PH3Q86",
                ],
            }
        )
        r = get_tickers_and_suffix(data)

        assert len(r.figi) == 5
        for a in [
            "BBG00NHX4J20",
            "BBG000PH98P0",
            "BBG000BVPV84",
            "BBG00XZP0LB4",
            "BBG009PH3Q86",
        ]:
            assert a in r[FIELD_FIGI].values

    def test_fetching_isin(self):
        data = pd.DataFrame(
            {"isin": ["US46222L1089", "IE00BYPLS672"], "currency": ["USD", "EUR"]}
        )
        r = get_tickers_and_suffix(data)

        assert len(r.figi) == 2
        for a in ["US46222L1089", "IE00BYPLS672"]:
            assert a in r[FIELD_ISIN].values

    def test_fetching_isin_not_found(self):
        data = pd.DataFrame(
            {"isin": ["US46222L1089", "US67020Y1001"], "currency": ["USD", "USD"]}
        )
        r = get_tickers_and_suffix(data)
        assert len(r.figi) == 1


if __name__ == "__main__":

    import os

    current_directory = os.getcwd()
    print("Current working dir: {}".format(current_directory))
    unittest.main()
    exit()
