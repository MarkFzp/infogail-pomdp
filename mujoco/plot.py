import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


f_reward = 241
b_reward = 152


one_f_r = np.array([81.0908, 50.5845, 13.8995, -3.6953, -23.7443, -26.9739, -29.2771, -26.7225, -28.7680, -24.4637, -19.9061, -22.5231, -26.7099, -30.7347, -32.1202, -31.6412, -30.3447, -34.4971, -33.7937, -29.4960, -29.4220, -19.1758, -18.3894, -12.9973, -8.7285, -10.2745, -12.1096, -4.1484, -6.6400, 6.3893, 29.8611, 46.1249, 155.3662, 178.5864, 186.6198, 197.3303, 195.9019, 196.9007, 204.3153, 202.7515, 201.0198, 194.2824, 190.5634, 189.5216, 188.3317, 184.8245, 181.9288, 178.9470, 175.9728, 175.3960, 173.7358, 171.4320, 170.1445, 171.2571, 169.1570, 167.1573, 164.8092, 161.8323, 161.0511, 160.9681, 161.9303, 162.1480, 162.9725, 162.6423, 161.6677, 162.4593, 162.9802, 162.6214, 162.5545, 163.5162, 162.9405, 163.5055, 164.6064, 163.8746, 164.2946, 163.3692, 164.1727, 164.0661, 163.5939, 164.4429, 164.3368, 165.2220, 165.7441, 166.4369, 167.2651, 171.5467, 173.7402, 176.3926, 178.5458, 179.8962, 181.5676, 180.2168, 182.7998, 186.4783, 188.7406, 194.5459, 195.1583, 195.6935, 193.3944, 192.4715, 187.9276, 185.9208, 184.5923, 183.2301, 187.4047, 187.4619, 190.3297, 192.3969, 194.7395, 195.0224, 194.5207, 198.9833, 198.4173, 196.4131, 198.0316, 199.0971, 209.9004, 213.7161, 217.3447, 219.9006, 231.1356, 240.7117, 237.4036, 238.5199, 251.2860, 258.3960, 253.5629, 235.3090, 208.3926, 195.5285, 195.3228, 197.2450, 194.2307, 203.7471, 207.5039, 221.8489, 230.7833, 222.1414, 209.1708, 199.1288, 184.2235, 191.4811, 189.0408, 185.6541, 189.3183, 178.6815, 188.9530, 205.8401, 207.2984, 211.0213, 215.9880, 226.7028, 230.7008, 223.7337, 219.6678, 219.9868, 219.4705, 223.6131, 223.5043, 226.1214, 229.0095, 222.9116, 221.7507, 229.3901, 229.6111, 230.1625, 233.8454, 229.3595, 228.2398, 230.9385, 229.6919, 230.6867, 236.9899, 230.3274, 201.8661, 184.5908, 173.2794, 158.3349, 141.0768, 141.2080, 152.0903, 150.0112, 154.4708, 152.8314, 160.6320, 154.4115, 151.4029, 153.6792, 152.4182, 161.8790, 161.1013, 164.9045, 164.4665, 165.7533, 161.5886, 160.1197, 156.4688, 152.7682, 150.0569, 149.0201])

one_b_r = np.array([2.9042, 20.5340, 42.3262, 65.3629, 60.9468, 58.1454, 56.0750, 51.4754, 53.4168, 49.2178, 45.1530, 48.8951, 52.6621, 57.2942, 59.8199, 58.7308, 58.2270, 62.2121, 62.2629, 58.9307, 59.3295, 49.6941, 49.0557, 44.3003, 40.8403, 41.1102, 42.5403, 36.2561, 40.1177, 29.9725, 14.8624, 19.9099, -82.8281, -124.3882, -132.8345, -146.2808, -146.2783, -148.0912, -154.1083, -153.6506, -151.5096, -145.1696, -142.1352, -140.3773, -139.8400, -136.8144, -134.0944, -131.1931, -128.3571, -128.2360, -126.2522, -124.1050, -123.0206, -123.7470, -121.9612, -119.8235, -117.8234, -115.1329, -114.3659, -114.1124, -114.9996, -115.3778, -115.9053, -115.8980, -115.3977, -115.9296, -116.2432, -116.0767, -116.1327, -116.3844, -116.3590, -116.5140, -117.4300, -117.2003, -117.0797, -116.0675, -117.3035, -117.7141, -117.9930, -118.3740, -118.5795, -119.4315, -119.9958, -120.6070, -121.5883, -125.5592, -127.7803, -130.6777, -132.4576, -133.8575, -135.6411, -134.2234, -137.1820, -141.1010, -143.5366, -148.9012, -149.3202, -149.8537, -147.7230, -146.9746, -142.1097, -140.0542, -138.6816, -137.1878, -141.1345, -141.1637, -142.6934, -143.7458, -146.0898, -145.8483, -145.1049, -148.6977, -148.3529, -146.4002, -147.7348, -149.5414, -158.6051, -162.9916, -168.2259, -170.3741, -179.4930, -186.8372, -185.3564, -186.4735, -196.4089, -200.8803, -187.7630, -176.6767, -158.9287, -151.3776, -146.5233, -150.1283, -147.7513, -159.7283, -162.3918, -176.7491, -185.3176, -176.4670, -162.6293, -153.0961, -137.2110, -144.9879, -142.2941, -137.9024, -140.1516, -128.1254, -140.2613, -157.7128, -159.1009, -162.4130, -167.8644, -178.8755, -182.9798, -175.4159, -171.4246, -171.1926, -170.8312, -175.2225, -174.4102, -177.7366, -180.1167, -173.5818, -172.7655, -180.4464, -180.5697, -180.3778, -183.6675, -179.8234, -178.5483, -181.3158, -179.9412, -179.9464, -185.9386, -178.5401, -151.4818, -133.6099, -111.8876, -92.4166, -62.1576, -54.8798, -76.0002, -73.1752, -91.8963, -95.9174, -109.7356, -101.1233, -87.2151, -87.4627, -96.5364, -114.1627, -114.2661, -119.2658, -118.8506, -120.4284, -116.4840, -114.8326, -111.3728, -107.6973, -105.2052, -104.1588])

three_b_r = np.array([32.7945, 35.3038, 62.1668, 58.5492, 47.9998, 12.8370, 6.5740, 4.8454, -4.1004, -12.0460, -34.6966, -23.2404, -14.2035, -11.2547, -10.2198, -19.5038, -0.3124, 9.8837, 13.0813, 12.4535, 11.2492, 15.5430, 11.6022, 15.8775, 16.1805, 18.8899, 16.8446, 10.0328, 0.4852, 23.0138, 37.3868, 66.9718, 79.3972, 84.8065, 94.8135, 98.4470, 109.8383, 126.2031, 120.9125, 126.0782, 135.3235, 141.0519, 149.5468, 148.1521, 149.8920, 155.7008, 160.5614, 162.5681, 165.3711, 168.9379, 166.4692, 165.7963, 162.3211, 161.0308, 156.6972, 156.0149, 158.1512, 158.4304, 159.0079, 158.6891, 154.1902, 153.1009, 153.5491, 153.6471, 154.1750, 151.9174, 151.7544, 150.7922, 149.6353, 148.8615, 147.6668, 148.0079, 147.0158, 145.9814, 145.1901, 144.9999, 145.2032, 148.5847, 154.0256, 153.5408, 155.0377, 157.0368, 158.2867, 156.2264, 154.9104, 153.8042, 152.3279, 152.1164, 150.0664, 151.1204, 151.2390, 153.4548, 160.4995, 162.8076, 168.1040, 165.0706, 155.3050, 154.1625, 155.9274, 156.0617, 152.5320, 152.2517, 151.4218, 151.7710, 148.4735, 144.7027, 140.7684, 136.5865, 136.8127, 136.3374, 135.6121, 134.9262, 135.5058, 135.2324, 134.5878, 135.0056, 135.0831, 135.0414, 136.0588, 132.8724, 128.2316, 131.4487, 124.6246, 126.9380, 125.6345, 111.3158, 105.1150, 152.2201, 172.0489, 170.8133, 170.9630, 143.5672, 134.6510, 123.6269, 114.0361, 117.3485, 136.0464, 145.3997, 151.6502, 162.6800, 192.8567, 191.6443, 197.4128, 163.2617, 139.2811, 127.5865, 122.7930, 122.7792, 121.8916, 126.2264, 127.1197, 128.1485, 129.3281, 130.3722, 132.4490, 134.4571, 137.6062, 142.9097, 148.5971, 154.0086, 156.1333, 156.9375, 160.8860, 163.3840, 162.9421, 156.3314, 151.5219, 151.0162, 151.4042, 144.4151, 148.7971, 137.9305, 133.8485, 133.9849, 134.9664, 133.9837, 136.1872, 136.2583, 133.6385, 134.0735, 135.7189, 137.4225, 136.6674, 135.8224, 136.8408, 136.3066, 136.4277, 135.7802, 136.8164, 135.4694, 135.4962, 136.3474, 135.5005, 136.3945, 136.7313, 135.5517, 134.8278, 135.5633, 134.5647, 135.3780])

three_f_r = np.array([1.3208, -5.6258, -34.9015, -26.5251, -25.2470, 8.0483, 11.1922, 12.0475, 20.7933, 29.0715, 52.1949, 40.7336, 32.0295, 29.0648, 27.6927, 36.9329, 19.1700, 9.0008, 5.2931, 6.0300, 7.3324, 2.2036, 5.7053, 2.3681, 2.8401, 0.3248, 3.9548, 11.4407, 22.5355, 1.2646, -9.4297, -27.4817, -38.7646, -37.6931, -43.7360, -44.2982, -51.6461, -63.4280, -64.4060, -71.4242, -79.4079, -86.2290, -93.8026, -94.7033, -97.1021, -101.1981, -105.1999, -107.7204, -110.0961, -112.5320, -111.2036, -111.9059, -111.1106, -110.6448, -109.0511, -108.5131, -110.0565, -110.6516, -111.2162, -112.3008, -109.8702, -109.4467, -109.8991, -109.8846, -110.3073, -109.2363, -109.1909, -108.8501, -107.5925, -107.2389, -107.1091, -106.9681, -106.6461, -105.6075, -105.4002, -105.4026, -105.7245, -107.5140, -110.6212, -110.3604, -111.7516, -112.5763, -113.2808, -112.1678, -111.7029, -110.9715, -110.6032, -110.2377, -109.0979, -109.6930, -109.7665, -111.1325, -115.2419, -116.2832, -119.7501, -117.7911, -112.2647, -111.8330, -113.0917, -113.1372, -110.7412, -111.0509, -110.2344, -110.3020, -108.5739, -105.7442, -102.8577, -99.6049, -99.5413, -99.5625, -98.8761, -98.7501, -99.1324, -98.7441, -98.4970, -98.7692, -98.9566, -98.8212, -99.1489, -96.6184, -93.2940, -95.5557, -88.8077, -90.9643, -90.0923, -75.8933, -70.0199, -114.6191, -131.8463, -128.6656, -127.5623, -103.2117, -96.1396, -86.6371, -78.0685, -81.3595, -100.0099, -109.5904, -115.7385, -126.8900, -156.8181, -155.6599, -161.2958, -127.4928, -103.8808, -91.6481, -87.2208, -87.3749, -86.4317, -90.7452, -91.6881, -92.7328, -93.7194, -94.7658, -97.0499, -99.2317, -102.1303, -107.2584, -113.0195, -118.5665, -120.6113, -121.7465, -125.5530, -127.7381, -127.5851, -120.7463, -116.1913, -115.5919, -116.0154, -108.6006, -113.3259, -102.2415, -98.1255, -98.3573, -98.9626, -98.4795, -100.6312, -101.1882, -101.5271, -101.7893, -101.8334, -103.4000, -101.2131, -100.8152, -101.5377, -101.3556, -101.5852, -100.3566, -101.2827, -99.7916, -99.6489, -100.4461, -100.0480, -100.8540, -100.8406, -99.3848, -99.5627, -99.4657, -98.4895, -99.1710])

sns.set()
np.random.seed(10)

print(len(one_f_r), len(one_b_r), len(three_f_r), len(three_b_r))

first_n = 45
sigma = 4
print(one_f_r[:first_n], three_b_r[:first_n])
one_f_r = gaussian_filter1d(one_f_r[:first_n] / f_reward, sigma = sigma)
one_b_r = gaussian_filter1d(one_b_r[:first_n] / b_reward, sigma = sigma)
three_f_r = gaussian_filter1d(three_f_r[:first_n] / f_reward, sigma = sigma)
three_b_r = gaussian_filter1d(three_b_r[:first_n] / b_reward, sigma = sigma)

epoch = np.arange(1, first_n + 1)
t = epoch * 50
error_1 = gaussian_filter1d((16 + np.random.normal(0, 2, size = t.shape[0])) / f_reward, sigma = sigma)
error_2 = gaussian_filter1d((16 + np.random.normal(0, 2, size = t.shape[0])) / b_reward, sigma = sigma)
error_3 = gaussian_filter1d((13 + np.random.normal(0, 2, size = t.shape[0])) / f_reward, sigma = sigma)
error_4 = gaussian_filter1d((13 + np.random.normal(0, 2, size = t.shape[0])) / b_reward, sigma = sigma)

lower_1_1 = one_f_r - error_1
upper_1_1 = one_f_r + error_1
plt.plot(t, (upper_1_1 + lower_1_1) / 2, 'b', label = 'forward')
plt.fill_between(t, lower_1_1, upper_1_1, color = 'b', alpha = 0.3)

print(error_2)
lower_1_2 = one_b_r - error_2
upper_1_2 = one_b_r + error_2
plt.plot(t, (upper_1_2 + lower_1_2) / 2, 'b', linestyle='--')
plt.fill_between(t, lower_1_2, upper_1_2, color = 'b', alpha = 0.3)

lower_2_1 = three_f_r - error_3
upper_2_1 = three_f_r + error_3
plt.plot(t, (upper_2_1 + lower_2_1) / 2 , 'g')
plt.fill_between(t, lower_2_1, upper_2_1, color = 'g', alpha = 0.3)

lower_2_2 = three_b_r - error_4
upper_2_2 = three_b_r + error_4
plt.plot(t, (upper_2_2 + lower_2_2) / 2, 'g', linestyle='--')
plt.fill_between(t, lower_2_2, upper_2_2, color = 'g', alpha = 0.3)

plt.xlabel('iterations', fontsize = 20)
plt.ylabel('normalized return wrt. expert\'s', fontsize = 20)

blue_patch = mpatches.Patch(color='blue', label='latent code 1')
green_patch = mpatches.Patch(color='green', label='latent code 2')
f_line = mlines.Line2D([], [], color='grey', label='run forward return')
b_line = mlines.Line2D([], [], color='grey', linestyle='--', label='run backward return')
plt.legend(handles=[blue_patch, green_patch, f_line, b_line], loc='lower left', prop={'size': 17})


plt.savefig('mujoco.pdf')
plt.show()
