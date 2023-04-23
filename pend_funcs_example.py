import pend_funcs as pf
import numpy as np

pend_exp_1 = pf.pend_experiment()

mag_mass = 0.00263
ramp_angle = np.arcsin(10.25/16)
ramp_dist = 0.4064
mag_offset = 1.5875
no_damp_values = [51/120, 52/120, 51/120, 50/120]
damp_values = [93/120, 95/120, 95/120]

pend_exp_1 = pf.pend_experiment()

pend_exp_1.magnet_mass_kg = mag_mass
pend_exp_1.ramp_angle_rad = ramp_angle
pend_exp_1.ramp_distance_m = ramp_dist
pend_exp_1.magnet_standoff_mm = mag_offset
pend_exp_1.t_final_damp_s_vec = damp_values
pend_exp_1.t_final_no_damp_s_vec = no_damp_values


print(pend_exp_1.calculate_friction_coefficient())

print(pend_exp_1.calculate_damping_coefficient())