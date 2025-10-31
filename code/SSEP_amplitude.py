# fig3f_g_ssep_amplitude.py
from utils import *
from scipy import signal

def search_p2p_value(sig, model='p2p'):
    if model == 'p2p':
        return np.max(sig) - np.min(sig)
    elif model == 'rms':
        return np.sqrt(np.mean(np.square(sig[220:])))
    elif model == 'hilbert_rms':
        _temp = signal.hilbert(sig)
        _temp = np.abs(_temp)
        _temp = _temp[220:]
        _temp = np.square(_temp)
        _temp = np.sqrt(np.mean(_temp))
        return _temp

electrodes = {
    "μECoG": "pedot",
    "10-μm aGel": "10um-hydrogel",
    "100-μm aGel": "100um-hydrogel"
}
currents = [50, 100, 200, 500, 1000]

all_amplitudes = {}
ref_amp = None

for name, folder in electrodes.items():
    amps = []
    for curr in currents:
        exp_path = find_experiment(BASE_PATH, folder, curr)  # 自定义查找函数
        data = load_and_preprocess(exp_path)
        data.update_map(build_1024_map())
        mask_ch = get_good_trial_mask(data)
        adaptive_rereference(data, build_1024_map())

        # 定位 high-gamma 9 通道
        gamma_map = data.frequency_band_map_view(70, 190, data_need=True, time_range=0.1)
        top9 = np.argsort(gamma_map.ravel())[-9:]
        chs = [build_1024_map().ravel()[i] for i in top9]

        # 提取平均信号
        sigs = [data.get_snr(ch, signal_need=True)[1] for ch in chs]
        avg_sig = np.mean(sigs, axis=0)
        p2p = search_p2p_value(avg_sig[200:300])  # 20–30 ms
        amps.append(p2p)

    all_amplitudes[name] = amps
    if name == "μECoG":
        ref_amp = amps[-1]  # 1 mA 幅值

# 归一化 + 绘图（略）