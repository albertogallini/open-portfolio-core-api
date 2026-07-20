import unittest

from oport.functions.functions_pattribution import is_root

import warnings


class TestErrors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Suppress DeprecationWarnings
        warnings.simplefilter("ignore", DeprecationWarning)

    def test_load_holdings_index_vs_index(self):
        from oport.openbb import obb

    
        output_folder = ".|tests|test_attribution_md|test_portfolios"

        r = obb.oport.portfolio_totals(
            portfolio_name="test111",
            filesystem_folder=output_folder,
            source="filesystem",
            classification="sector",
            start_date="10-6-2025",
            end_date="10-10-2025",
        )
        self.assertTrue("FAILED" in r.results[1])
        self.assertTrue("Missing the input stream" in r.results[1])

if __name__ == "__main__":
    import os

    current_directory = os.getcwd()
    print("Current working dir: {}".format(current_directory))
    unittest.main()

# %%
