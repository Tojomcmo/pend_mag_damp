import mag_damp_funcs as pf
import numpy as np

pend_exp_1 = pf.pend_experiment()
num_experiments = 5

description = [['round lg 1 mag 10.25 height'],
               ['round lg 1 mag 15 height'],
               ['rect 1 mag 10.75 height'],
               ['rect 1 mag 14.5 height'],
               ['rect 1 mag 14.5 height no plastic']]

mag_mass = [0.00263,
            0.00263,
            0.00423,
            0.00423,
            0.00423]
ramp_angle = [np.arcsin(10.25/16), 
              np.arcsin(15/16),
              np.arcsin(10.75/16),
              np.arcsin(14.5/16),
              np.arcsin(14.5/16)]
ramp_dist = [0.4064,
             0.4064,
             0.4064,
             0.4064,
             0.4064]
mag_offset = [1.5875,
              1.5875,
              1.5875,
              1.5875,
              0]
no_damp_values = [[51/120, 52/120, 51/120, 50/120],
                  [38/120, 36/120, 37/120, 38/120, 39/120, 36/120],
                  [46/120, 49/120, 48/120, 50/120, 47/120],
                  [37/120, 37/120, 37/120],
                  [37/120, 37/120, 37/120]]
damp_values = [[93/120, 95/120, 95/120], 
               [54/120, 53/120, 60/120, 57/120, 56/120, 57/120],
               [153/120, 158/120, 154/120],
               [100/120, 110/120, 112/120, 109/120],
               [250/120, 270/120, 260/120]]

exp_3_xdot_ss = 5/(45/120) * 25.4/1000
exp_4_xdot_ss = 5/(32/120) * 25.4/1000

pend_exps       = [None] * num_experiments
friction_coeffs = [None] * num_experiments
damping_coeffs  = [None] * num_experiments

for idx in range(num_experiments):
    pend_exps[idx] = pf.pend_experiment()
    pend_exps[idx].description           = description[idx]    
    pend_exps[idx].magnet_mass_kg        = mag_mass[idx]
    pend_exps[idx].ramp_angle_rad        = ramp_angle[idx]
    pend_exps[idx].ramp_distance_m       = ramp_dist[idx]
    pend_exps[idx].magnet_standoff_mm    = mag_offset[idx]
    pend_exps[idx].t_final_no_damp_s_vec = no_damp_values[idx]
    pend_exps[idx].t_final_damp_s_vec    = damp_values[idx]

    friction_coeffs[idx] = pend_exps[idx].calculate_friction_coefficient()
    damping_coeffs[idx]  = pend_exps[idx].calculate_damping_coefficient()


print('friction coefficients:  ', friction_coeffs)

print('damping coefficients:  ', damping_coeffs)

print('exp_3_damp_coeff_from_ss', pend_exps[2].calc_steady_state_damping_coefficient(exp_3_xdot_ss))
print('exp_4_damp_coeff_from_ss', pend_exps[3].calc_steady_state_damping_coefficient(exp_4_xdot_ss))
print('exp_4_override_mu', pend_exps[3].calculate_damping_coefficient(mu_k = 0.23))