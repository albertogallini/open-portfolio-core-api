import unittest
from oport.functions.linking import carino_linking_return_split
import warnings


class TestLinking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Suppress DeprecationWarnings
        warnings.simplefilter("ignore", DeprecationWarning)

    def test_linking_additivity(self):
        import numpy as np

        r_c = np.array([0.9867407911, 0.9856957118, 1.014072106, 1.023417213, 1, 1])
        r_l = np.array([1, 1.000910078, 1.004698909, 0.9968273219, 1, 1])
        # r_tot = np.array([ 0.9867407911, 0.9866057894,1.018771015, 1.020244535 ,1,1])
        r_tot = (r_c - 1 + r_l - 1) + 1
        print("r_tot {}".format(r_tot))

        R_c = carino_linking_return_split(r_c, r_tot, np.prod(r_tot))
        R_l = carino_linking_return_split(r_l, r_tot, np.prod(r_tot))

        print("R_c {}".format(R_c))
        print("R_l {}".format(R_l))
        print("R_tot {} - {} ".format(np.prod(r_tot), R_l + R_c - 1))

        self.assertTrue(R_l + R_c - 1 - np.prod(r_tot) < 0.0001)

        acc_lcl_ccy_r = [1.0]
        acc_ccy_r = [1.0]
        acc_r = [1.0]
        for i in range(0, len(r_c)):
            print(r_tot[i])
            acc_total_return_i = acc_r[i] * (r_tot[i])
            acc_r.append(acc_total_return_i)
            acc_lcl_ccy_return_i = carino_linking_return_split(
                r_l[: i + 1], r_tot[: i + 1], acc_total_return_i
            )
            acc_lcl_ccy_r.append(acc_lcl_ccy_return_i)
            acc_ccy_return_i = carino_linking_return_split(
                r_c[: i + 1], r_tot[: i + 1], acc_total_return_i
            )
            acc_ccy_r.append(acc_ccy_return_i)

        print(acc_r)
        print(np.add(acc_lcl_ccy_r, acc_ccy_r) - 1)

        for i in range(0, len(acc_r)):
            self.assertTrue(
                acc_r[i] - np.add(acc_lcl_ccy_r, acc_ccy_r)[i] - 1 < 0.000001
            )


if __name__ == "__main__":
    import os

    current_directory = os.getcwd()
    print("Current working dir: {}".format(current_directory))
    unittest.main()
