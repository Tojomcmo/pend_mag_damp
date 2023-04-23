import numpy as np

class pend_experiment(object):
    def __init__(self, 
                 magnet_mass_kg = None, 
                 ramp_angle_rad = None, 
                 ramp_distance_m = None,
                 magnet_standoff_mm = None, 
                 t_final_no_damp_s_vec = [], 
                 t_final_damp_s_vec = []):
        
        self.magnet_mass_kg        = magnet_mass_kg
        self.ramp_angle_rad        = ramp_angle_rad
        self.ramp_distance_m       = ramp_distance_m
        self.magnet_standoff_mm    = magnet_standoff_mm
        self.t_final_no_damp_s_vec = t_final_no_damp_s_vec
        self.t_final_damp_s_vec    = t_final_damp_s_vec
        self.gravity_m_s2          = 9.81

        return
    
    def get_avg_t_final(self, test_type):
        if test_type == 'no_damp':
            avg_time = sum(self.t_final_no_damp_s_vec) / len(self.t_final_no_damp_s_vec)
        elif test_type == 'damp':
            avg_time = sum(self.t_final_damp_s_vec) / len(self.t_final_damp_s_vec)
        else:
            NameError
        return avg_time
        
    def calculate_friction_coefficient(self):
        xf = self.ramp_distance_m
        th = self.ramp_angle_rad
        tf = self.get_avg_t_final('no_damp')
        mu_k = (-2 * xf) / (9.81 * tf**2) + np.sin(th) / np.cos(th)
        return mu_k
    
    def calculate_damping_coefficient(self):
        xf   = self.ramp_distance_m
        m    = self.magnet_mass_kg
        g    = self.gravity_m_s2
        th   = self.ramp_angle_rad
        tf   = self.get_avg_t_final('damp')
        mu_k = self.calculate_friction_coefficient()
        a    = g * (np.sin(th) - mu_k * np.cos(th))

        x_f_check = xf + 1 
        b         = 0

        while x_f_check >= xf:
            b = b + 1e-5
            x_f_check = (a * m**2)/(b**2) * np.exp((-b)/(m)*tf) + (a * m)/(b) * tf - (a * m**2)/(b**2)
        return b