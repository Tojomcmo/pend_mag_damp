import unittest
import mag_damp_funcs as pf



class pend_experiment_class_tests(unittest.TestCase):
    def test_pend_experiment_class_provides_correct_averages(self):

        no_damp_values = [0.09, 0.1, 0.11]
        damp_values    = [0.09, 0.1, 0.11, 0.12, 0.13]
        pend_exp       = pf.pend_experiment(t_final_damp_s_vec = damp_values, t_final_no_damp_s_vec = no_damp_values)
        self.assertAlmostEqual(pend_exp.get_avg_t_final('damp'),       0.11)
        self.assertAlmostEqual(pend_exp.get_avg_t_final('no_damp'),    0.10)



if __name__ == '__main__':
    unittest.main()