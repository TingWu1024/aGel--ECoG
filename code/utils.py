# utils.py
import os
import numpy as np
from WuLab_utils.data_dealer import WuLab_data
from WuLab_utils import electrode_map


def build_1024_map():
    data_map = electrode_map.get_1024map()
    port_names = ['A', 'B', 'C', 'D', 'H', 'G', 'F', 'E']
    for k in range(8):
        data_map[port_names[k]] += k * 128
    map_1024 = np.zeros((32, 32), dtype=int)
    for k in range(8):
        row = (k // 4) * 16
        col = (k % 4) * 8
        map_1024[row:row + 16, col:col + 8] = data_map[port_names[k]]
    import scipy.ndimage
    map_1024 = scipy.ndimage.rotate(map_1024, 270)
    map_1024 = np.flipud(map_1024)
    return map_1024


def load_and_preprocess(exp_path):
    data = None
    for file in os.listdir(exp_path):
        if file.endswith('.rhd'):
            temp = WuLab_data(os.path.join(exp_path, file))
            temp.band_pass_filter(4, 1000)
            temp.down_sample(2000)
            temp.iir_comb_filter_sos(50, quality=30)
            data = temp if data is None else data + temp
    return data


def get_good_trial_mask(data, threshold_multiple=1.5):
    rates = []
    for ch in range(data.__signal_data.shape[0]):
        rate = data.get_snr(ch, threshold_time_range=0.5, threshold_multiple=threshold_multiple,
                            time_range=0.5, mask_bad_trial=True, sig_rate_need=True)
        rates.append(rate)
    rates = np.array(rates)
    bad_trial = np.where(rates < 0.1)[0]
    bad_imp = data.get_bad_channels(imp=1e7)[0]
    return np.unique(np.concatenate([bad_imp, bad_trial]))


def adaptive_rereference(data, map_1024):
    # Step 1: 初始 SNR map
    snr_map = data.get_snr_map(threshold_time_range=0.2, threshold_multiple=2, time_range=0.1,
                               model='hilbert_sig', need_data=True)
    # Step 2: 2D Gaussian 拟合
    from scipy import optimize
    xx, yy = np.meshgrid(np.arange(32), np.arange(32))

    def gauss(xy, a, x0, y0, sigma, c):
        x, y = xy
        return a * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) + c

    popt, _ = optimize.curve_fit(gauss, np.vstack([xx.ravel(), yy.ravel()]), snr_map.ravel())
    snr_gauss = gauss(np.vstack([xx.ravel(), yy.ravel()]), *popt).reshape(32, 32)
    # Step 3: 选 SNR 最低 100 通道
    ref_idx = np.argpartition(snr_gauss.ravel(), 100)[:100]
    ref_chs = map_1024.ravel()[ref_idx]
    ref_chs = ref_chs[~np.isnan(ref_chs)].astype(int)
    data.re_reference(chs=ref_chs)
