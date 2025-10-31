# fig3i_spatial_resolution.py
from utils import *
from scipy import optimize

exp_path = "path/to/silence_exp"
data = load_and_preprocess(exp_path)
map_1024 = build_1024_map()
data.update_map(map_1024, new_distance=0.2)  # 200 μm = 0.2 mm
data.time_cut(20, 40)  # 静息态中的一段
mask_ch = get_good_trial_mask(data)
good_chs = np.setdiff1d(map_1024.ravel(), mask_ch)
data.re_reference(chs=good_chs)

# 计算 coherence
coherence = data.spatial_resolution(
    model='coherence',
    freq_band=[70, 190],
    signal_segment=5,
    data_need=True
)

# 拟合 λ
distances = coherence[:, 0]
coherences = coherence[:, 1]
unique_d = np.unique(distances)
mean_coh = [np.mean(coherences[distances == d]) for d in unique_d]


def exp_decay(x, a, lam, c):
    return a * np.exp(-x / lam) + c


popt, _ = optimize.curve_fit(exp_decay, unique_d, mean_coh)
lambda_fit = popt[1]
