# fig3h_snr_maps.py
from utils import *

exp_path = "path/to/exp_200ua"
data = load_and_preprocess(exp_path)
map_1024 = build_1024_map()
data.update_map(map_1024)
mask_ch = get_good_trial_mask(data)
adaptive_rereference(data, map_1024)

bands = [(4,8), (8,12), (12,30), (30,70), (70,190)]
for low, high in bands:
    # 先滤波
    original = data.__signal_data.copy()
    data.band_pass_filter(low, high)
    # 再计算 SNR map
    snr_map = data.get_snr_map(
        threshold_time_range=0.5,
        threshold_multiple=1.5,
        time_range=0.1,
        model='hilbert_sig',
        mask_ch=mask_ch,
        need_data=True
    )
    # 保存/绘图
    data.__signal_data = original  # 恢复