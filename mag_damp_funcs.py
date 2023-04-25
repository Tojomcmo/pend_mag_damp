import numpy as np
import jax.numpy as jnp
import misc_funcs as mf
import matplotlib.pyplot as plt

class pend_experiment(object):
    def __init__(self, 
                 description           = None,
                 magnet_mass_kg        = None, 
                 ramp_angle_rad        = None, 
                 ramp_distance_m       = None,
                 magnet_standoff_mm    = None, 
                 t_final_no_damp_s_vec = [], 
                 t_final_damp_s_vec    = []):
        
        self.description           = description
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
        mu_k = mf.calculate_mu_k_from_final_time(self.ramp_distance_m, 
                                                 self.get_avg_t_final('no_damp'), 
                                                 self.ramp_angle_rad,
                                                 self.gravity_m_s2)
        return mu_k
    
    def calculate_damping_coefficient(self, mu_k = None, eps = 1e-6, b_init = 0.01):
        xf   = self.ramp_distance_m
        m    = self.magnet_mass_kg
        g    = self.gravity_m_s2
        th   = self.ramp_angle_rad
        tf   = self.get_avg_t_final('damp')
        if mu_k == None:
            mu_k = self.calculate_friction_coefficient()     
        a             = mf.calculate_init_ramp_acc(th, mu_k, g)
        
        b_test        = b_init
        xf_wrt_b_func = lambda b: (mf.calculate_x_final_with_damping(b,a,m,tf) - xf)   
        xf_test       = xf_wrt_b_func(b_test)
        iter          = 0
        while jnp.abs(xf_test) > eps:
            # equation for x_f_check derived from diff eq describing ramp problem with linear damping
            # x_f_check = (a * m**2)/(b**2) * jnp.exp((-b)/(m)*tf) + (a * m)/(b) * tf - (a * m**2)/(b**2)
            b_test = mf.NRmethod(xf_wrt_b_func, b_test) 
            xf_test = xf_wrt_b_func(b_test)
            iter += 1
        return b_test, iter
    
    def graph_x_final_vs_damping_coefficient(self, b_vec = jnp.linspace(0.001, 1, 100), mu_k = None):
        m    = self.magnet_mass_kg
        g    = self.gravity_m_s2
        th   = self.ramp_angle_rad
        tf   = self.get_avg_t_final('damp')
        if mu_k == None:
            mu_k = self.calculate_friction_coefficient()    
        a    = mf.calculate_init_ramp_acc(th, mu_k, g)
        x_final = [None] * len(b_vec)

        for idx, ib in enumerate(b_vec):
            # equation for x_f_check derived from diff eq describing ramp problem with linear damping
            x_final[idx] = mf.calculate_x_final_with_damping(ib,a,m,tf)

        plt.plot(b_vec, x_final)
        plt.show()
        return None
    
    def calc_steady_state_damping_coefficient(self, x_dot_measured_m_s):
        th   = self.ramp_angle_rad
        g    = self.gravity_m_s2
        m    = self.magnet_mass_kg
        mu_k = self.calculate_friction_coefficient()
        a    = mf.calculate_init_ramp_acc(th, mu_k, g)
        b    = (m * g)*(jnp.sin(th) - mu_k * jnp.cos(th))/x_dot_measured_m_s
        return b
