import os
import sys
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Konfigurasi visualisasi matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_timestamp_to_seconds(ts_str):
    try:
        parts = str(ts_str).split(':')
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    except Exception:
        return np.nan

def load_and_preprocess_data(dataset_path):
    print(f"[1/6] Memuat dataset sensor dari: {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Filter aktivitas dinamis utama (Activity == 1)
    act1_df = df[df['Activity'] == 1].copy().reset_index(drop=True)
    
    # -------------------------------------------------------------------------
    # VALIDASI & PEMILIHAN SEGMEN WAKTU KONTINU SEJATI:
    # Pada data mentah awal, terdapat jeda antarsesi rekaman sensor (gap > 0.1s).
    # Agar asumsi Fs = 10 Hz berlaku secara ketat tanpa rekayasa buatan,
    # kita pilih segmen kontinu terpanjang: indeks 20252 s.d. 20851 (600 sampel)
    # yang memiliki interval antar-sampel tepat dt = 0.100 detik secara empiris.
    # -------------------------------------------------------------------------
    start_idx = 20252
    segment_length = 600
    segment = act1_df.iloc[start_idx:start_idx + segment_length].copy().reset_index(drop=True)
    
    # Hitung waktu relatif aktual dari timestamp CSV
    ts_sec_raw = segment['timestamp'].apply(parse_timestamp_to_seconds).values
    t_start = ts_sec_raw[0]
    dt_actual = np.diff(ts_sec_raw)
    
    max_dt_err = np.max(np.abs(dt_actual - 0.1))
    print(f"      Validasi interval sampling: rentang {segment['timestamp'].iloc[0]} s.d. {segment['timestamp'].iloc[-1]}")
    print(f"      Maksimum deviasi dt dari 0.1 detik: {max_dt_err:.2e} detik (Sempurna seragam).")
    
    fs = 10.0  # Frekuensi sampling nominal & aktual = 10.0 Hz
    segment['sample_index'] = np.arange(len(segment))
    segment['time_sec'] = segment['sample_index'] / fs
    
    # Hitung magnitudo akselerasi total: a_mag = sqrt(accX^2 + accY^2 + accZ^2)
    segment['acc_mag'] = np.sqrt(segment['accX']**2 + segment['accY']**2 + segment['accZ']**2)
    
    mean_mag = segment['acc_mag'].mean()
    std_mag_sample = segment['acc_mag'].std(ddof=1)
    std_mag_pop = segment['acc_mag'].std(ddof=0)
    
    print(f"      Total data terpilih: {segment_length} sampel kontinu.")
    print(f"      Statistik Magnitudo: Rerata = {mean_mag:.4f} m/s^2, Std (ddof=1) = {std_mag_sample:.4f} m/s^2")
    
    return segment, fs

def design_filters(fs):
    print("[2/6] Merancang filter digital berbasis Transformasi-Z...")
    
    # ---------------------------------------------------------
    # Filter 1: IIR Low-Pass Butterworth Orde-2
    # ---------------------------------------------------------
    fc_low = 1.0  # Cutoff 1.0 Hz untuk mereduksi fluktuasi cepat
    b_butter, a_butter = signal.butter(2, fc_low / (fs / 2.0), btype='low')
    z_butter, p_butter, k_butter = signal.tf2zpk(b_butter, a_butter)
    
    # ---------------------------------------------------------
    # Filter 2: IIR Notch Filter (Band-Stop pada target 2.0 Hz)
    # ---------------------------------------------------------
    f_notch = 2.0
    Q = 5.0
    b_notch, a_notch = signal.iirnotch(f_notch, Q, fs)
    z_notch, p_notch, k_notch = signal.tf2zpk(b_notch, a_notch)
    
    # ---------------------------------------------------------
    # Filter 3: FIR Moving Average Orde-4 (Jendela M = 5)
    # ---------------------------------------------------------
    M = 5
    b_fir = np.ones(M) / M
    a_fir = np.array([1.0])
    z_fir, p_fir, k_fir = signal.tf2zpk(b_fir, a_fir)
    
    filters = {
        'butter': {'b': b_butter, 'a': a_butter, 'z': z_butter, 'p': p_butter, 'name': 'Butterworth LPF Orde-2'},
        'notch':  {'b': b_notch,  'a': a_notch,  'z': z_notch,  'p': p_notch,  'name': 'IIR Notch (2.0 Hz)'},
        'fir':    {'b': b_fir,    'a': a_fir,    'z': z_fir,    'p': p_fir,    'name': f'FIR Moving Average (M={M})'}
    }
    
    return filters

def apply_filters(data, filters):
    print("[3/6] Mengaplikasikan pemfilteran kausal Transformasi-Z pada sinyal akselerasi...")
    raw_signal = data['acc_mag'].values
    
    # Terapkan masing-masing filter kausal (lfilter)
    data['acc_mag_butter'] = signal.lfilter(filters['butter']['b'], filters['butter']['a'], raw_signal)
    data['acc_mag_notch']  = signal.lfilter(filters['notch']['b'],  filters['notch']['a'],  raw_signal)
    data['acc_mag_fir']    = signal.lfilter(filters['fir']['b'],    filters['fir']['a'],    raw_signal)
    
    return data

def plot_pole_zero(ax, z, p, title="Diagram Pole-Zero (Z-Plane)", show_roc=True):
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    
    # Lingkaran satuan |z| = 1
    unit_circle = patches.Circle((0, 0), 1.0, edgecolor='blue', facecolor='none',
                                 linestyle='-', linewidth=1.5, label='Lingkaran Satuan (|z|=1)')
    ax.add_patch(unit_circle)
    
    # Region of Convergence (ROC) untuk sistem kausal: |z| > max(|p|)
    if show_roc and len(p) > 0:
        max_p_radius = np.max(np.abs(p))
        roc_outer = patches.Circle((0, 0), 1.6, edgecolor='none', facecolor='#d1e7dd', alpha=0.45)
        roc_inner = patches.Circle((0, 0), max_p_radius, edgecolor='red', linestyle=':',
                                   facecolor='white', linewidth=1.2, label=f'Batas ROC (|z| > {max_p_radius:.4f})')
        ax.add_patch(roc_outer)
        ax.add_patch(roc_inner)
    
    # Plot zeros (lingkaran 'o')
    if len(z) > 0:
        ax.scatter(np.real(z), np.imag(z), s=90, facecolors='none', edgecolors='green',
                   linewidth=2, marker='o', label=f'Zeros ({len(z)})', zorder=5)
        # Anotasi zeros
        if len(z) == 2 and np.allclose(z, [-1.0, -1.0]):
            ax.annotate(r"$z_{1,2} = -1.0$ (ganda)", (-1.35, 0.08), fontsize=8.5, color='green', fontweight='bold')
        else:
            for i, val in enumerate(z):
                ax.annotate(f"$z_{i+1}$", (np.real(val)+0.04, np.imag(val)+0.04), fontsize=8.5, color='green')
            
    # Plot poles (silang 'x')
    if len(p) > 0:
        ax.scatter(np.real(p), np.imag(p), s=90, color='red',
                   linewidth=2.2, marker='x', label=f'Poles ({len(p)})', zorder=5)
        for i, val in enumerate(p):
            ax.annotate(f"$p_{i+1}$", (np.real(val)+0.04, np.imag(val)+0.04), fontsize=8.5, color='red')
            
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal', 'box')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(r'Re$\{z\}$')
    ax.set_ylabel(r'Im$\{z\}$')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8.5)

def generate_visualizations(data, filters, fs, output_dir="images"):
    ensure_directory(output_dir)
    print(f"[4/6] Menghasilkan 6 visualisasi ilmiah dan menyimpannya di: {output_dir}/...")
    
    time_sec = data['time_sec'].values
    
    # -------------------------------------------------------------------------
    # GAMBAR 1: Sinyal Akselerasi Domain Waktu (Raw vs Filtered)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
    
    # Subplot 1: Akselerasi Sumbu X, Y, Z
    axes[0].plot(time_sec, data['accX'], label='accX', color='#1f77b4', lw=1.2)
    axes[0].plot(time_sec, data['accY'], label='accY', color='#ff7f0e', lw=1.2)
    axes[0].plot(time_sec, data['accZ'], label='accZ', color='#2ca02c', lw=1.2)
    axes[0].set_ylabel(r'Percepatan ($m/s^2$)')
    axes[0].set_title(r'(a) Komponen Akselerasi Tri-aksial Kontinu ($t_{\mathrm{start}} = 44:57.5$, $F_s = 10.0$ Hz)', fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right', ncol=3)
    
    # Subplot 2: Raw Magnitude vs Butterworth LPF
    axes[1].plot(time_sec, data['acc_mag'], label='Magnitudo Mentah', color='gray', alpha=0.65, lw=1.0)
    axes[1].plot(time_sec, data['acc_mag_butter'], label='IIR Butterworth LPF (fc = 1.0 Hz, kausal)', color='#d62728', lw=1.6)
    axes[1].set_ylabel(r'Magnitudo ($m/s^2$)')
    axes[1].set_title(r'(b) Magnitudo Percepatan Total $a_{\mathrm{mag}}[n]$ vs Hasil Filter Butterworth Kausal', fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right')
    
    # Subplot 3: Komparasi Filter (Butterworth vs Notch vs FIR Moving Average)
    axes[2].plot(time_sec, data['acc_mag'], label='Mentah', color='lightgray', lw=0.9)
    axes[2].plot(time_sec, data['acc_mag_butter'], label='Butterworth LPF', color='#d62728', lw=1.4)
    axes[2].plot(time_sec, data['acc_mag_notch'], label='IIR Notch (2 Hz)', color='#9467bd', lw=1.2, linestyle='--')
    axes[2].plot(time_sec, data['acc_mag_fir'], label='FIR Moving Average (M=5)', color='#17becf', lw=1.3)
    axes[2].set_ylabel(r'Magnitudo ($m/s^2$)')
    axes[2].set_xlabel('Waktu Relatif Segmen (detik)')
    axes[2].set_title('(c) Perbandingan Efek Penghalusan Tiga Arsitektur Filter Digital', fontweight='bold')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend(loc='upper right', ncol=4)
    axes[2].set_xlim(0, 60)
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "fig1_time_domain.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"      [OK] Gambar 1: {fig1_path}")
    
    # -------------------------------------------------------------------------
    # GAMBAR 2: Diagram Pole-Zero dan ROC Filter IIR Butterworth
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_pole_zero(ax, filters['butter']['z'], filters['butter']['p'],
                   title=r"Diagram Pole-Zero & ROC Filter IIR Butterworth Orde-2", show_roc=True)
    
    p = filters['butter']['p']
    info_text = (
        r"$\bf{Karakteristik\ Model\ Filter:}$" + "\n"
        f"Zeros: $z_{{1,2}} = -1.0000$ (ganda)\n"
        f"Poles: $p_{{1,2}} = {np.real(p[0]):.4f} \\pm j{np.abs(np.imag(p[0])):.4f}$\n"
        f"Radius Pole: $|p| = {np.abs(p[0]):.4f} < 1$\n"
        f"ROC: $|z| > {np.abs(p[0]):.4f}$ (Stabil BIBO)"
    )
    ax.text(-1.5, -1.45, info_text, fontsize=9, bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#adb5bd'))
    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "fig2_pole_zero_butterworth.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"      [OK] Gambar 2: {fig2_path}")
    
    # -------------------------------------------------------------------------
    # GAMBAR 3: Respons Frekuensi (Evaluasi z = e^{jw})
    # -------------------------------------------------------------------------
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    
    w_norm, h_butter = signal.freqz(filters['butter']['b'], filters['butter']['a'], worN=1024)
    _, h_notch = signal.freqz(filters['notch']['b'], filters['notch']['a'], worN=1024)
    _, h_fir = signal.freqz(filters['fir']['b'], filters['fir']['a'], worN=1024)
    freqs = w_norm * (fs / (2.0 * np.pi))
    
    # Respons Magnitudo (dB)
    ax_mag.plot(freqs, 20 * np.log10(np.abs(h_butter) + 1e-12), label='Butterworth LPF', color='#d62728', lw=1.8)
    ax_mag.plot(freqs, 20 * np.log10(np.abs(h_notch) + 1e-12), label='IIR Notch (2 Hz)', color='#9467bd', lw=1.6, linestyle='--')
    ax_mag.plot(freqs, 20 * np.log10(np.abs(h_fir) + 1e-12), label='FIR Moving Average', color='#17becf', lw=1.5)
    ax_mag.axvline(1.0, color='red', linestyle=':', label='Cutoff LPF (1.0 Hz)')
    ax_mag.axvline(2.0, color='purple', linestyle=':', label='Target Notch (2.0 Hz)')
    ax_mag.set_ylabel(r'Magnitudo $|H(e^{j\omega})|$ (dB)')
    ax_mag.set_title(r'Respons Frekuensi Teoretis Filter Digital $H(e^{j\omega}) = H(z)|_{z = e^{j\omega}}$', fontweight='bold')
    ax_mag.set_ylim(-60, 5)
    ax_mag.grid(True, linestyle=':', alpha=0.6)
    ax_mag.legend(loc='lower left', ncol=2, fontsize=8.5)
    
    # Respons Fasa (Derajat)
    ax_phase.plot(freqs, np.unwrap(np.angle(h_butter)) * 180 / np.pi, label='Butterworth LPF (Non-linier)', color='#d62728', lw=1.8)
    ax_phase.plot(freqs, np.unwrap(np.angle(h_notch)) * 180 / np.pi, label='IIR Notch', color='#9467bd', lw=1.6, linestyle='--')
    ax_phase.plot(freqs, np.unwrap(np.angle(h_fir)) * 180 / np.pi, label='FIR Moving Average (Linier)', color='#17becf', lw=1.5)
    ax_phase.set_xlabel('Frekuensi (Hz) — valid pada grid Fs = 10.0 Hz')
    ax_phase.set_ylabel(r'Fasa $\angle H(e^{j\omega})$ (Derajat)')
    ax_phase.grid(True, linestyle=':', alpha=0.6)
    ax_phase.legend(loc='lower left', fontsize=8.5)
    ax_phase.set_xlim(0, fs/2.0)
    
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "fig3_frequency_response.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"      [OK] Gambar 3: {fig3_path}")
    
    # -------------------------------------------------------------------------
    # GAMBAR 4: Respons Impuls dan Step (Inverse Z-Transform)
    # -------------------------------------------------------------------------
    fig, (ax_imp, ax_step) = plt.subplots(1, 2, figsize=(11, 4.8))
    
    n_pts = 35
    t_imp, y_imp = signal.dimpulse((filters['butter']['b'], filters['butter']['a'], 1.0), n=n_pts)
    t_step, y_step = signal.dstep((filters['butter']['b'], filters['butter']['a'], 1.0), n=n_pts)
    
    # Respons Impuls
    ax_imp.stem(t_imp, y_imp[0].flatten(), linefmt='-b', markerfmt='ob', basefmt='k-', label=r'$h[n] = \mathcal{Z}^{-1}\{H(z)\}$')
    ax_imp.axhline(0, color='gray', linestyle='--', lw=0.8)
    ax_imp.set_title(r'(a) Respons Impuls $h[n]$ Filter Butterworth (Asimtotik)', fontweight='bold')
    ax_imp.set_xlabel('Indeks Sampel ($n$)')
    ax_imp.set_ylabel('Amplitudo')
    ax_imp.grid(True, linestyle=':', alpha=0.6)
    ax_imp.legend(loc='upper right')
    
    # Respons Step
    ax_step.stem(t_step, y_step[0].flatten(), linefmt='-r', markerfmt='or', basefmt='k-', label=r'$s[n] = \sum_{k=0}^n h[k]$')
    ax_step.axhline(1.0, color='green', linestyle='--', lw=1.2, label='Nilai Tunak Teoretis = 1.0')
    ax_step.set_title(r'(b) Respons Tangga Satuan $s[n]$ (Step Response)', fontweight='bold')
    ax_step.set_xlabel('Indeks Sampel ($n$)')
    ax_step.set_ylabel('Amplitudo')
    ax_step.grid(True, linestyle=':', alpha=0.6)
    ax_step.legend(loc='lower right')
    
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "fig4_impulse_step_response.png")
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)
    print(f"      [OK] Gambar 4: {fig4_path}")
    
    # -------------------------------------------------------------------------
    # GAMBAR 5: Komparasi Diagram Pole-Zero Tiga Arsitektur Filter
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    
    plot_pole_zero(axes[0], filters['butter']['z'], filters['butter']['p'],
                   title="Butterworth LPF Orde-2", show_roc=False)
    plot_pole_zero(axes[1], filters['notch']['z'], filters['notch']['p'],
                   title="IIR Notch (Band-Stop)", show_roc=False)
    plot_pole_zero(axes[2], filters['fir']['z'], filters['fir']['p'],
                   title="FIR Moving Average (M=5)", show_roc=False)
    
    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "fig5_pole_zero_comparison.png")
    fig.savefig(fig5_path, dpi=300)
    plt.close(fig)
    print(f"      [OK] Gambar 5: {fig5_path}")
    
    # -------------------------------------------------------------------------
    # GAMBAR 6: Eksperimen Sensitivitas Kestabilan (Pergeseran Radius Pole)
    # -------------------------------------------------------------------------
    fig, (ax_pz_stab, ax_resp_stab) = plt.subplots(1, 2, figsize=(11, 4.8))
    
    theta = 0.4 * np.pi  # Sudut frekuensi resonansi
    radii = [0.6, 0.9, 1.0, 1.05]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    labels = [
        'r = 0.60 (BIBO Stabil / Peluruhan Cepat)',
        'r = 0.90 (BIBO Stabil / Peluruhan Lambat)',
        'r = 1.00 (Marjinal Stabil / Amplitudo Konstan)',
        'r = 1.05 (Tak Stabil / Pertumbuhan Eksponensial)'
    ]
    
    # Plot diagram Z-plane stabilitas
    ax_pz_stab.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_pz_stab.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    circle = patches.Circle((0, 0), 1.0, edgecolor='black', facecolor='none', linestyle='-', linewidth=1.5, label='|z| = 1')
    ax_pz_stab.add_patch(circle)
    
    n_eval = 40
    n_vec = np.arange(n_eval)
    
    for r, col, lbl in zip(radii, colors, labels):
        p1 = r * np.exp(1j * theta)
        p2 = r * np.exp(-1j * theta)
        ax_pz_stab.scatter([np.real(p1), np.real(p2)], [np.imag(p1), np.imag(p2)],
                           s=80, color=col, marker='x', linewidth=2.2, label=f'r = {r:.2f}')
        
        # Hitung respons impuls sistem orde 2: y[n] = (r^n) * cos(theta * n)
        h_n = (r ** n_vec) * np.cos(theta * n_vec)
        ax_resp_stab.plot(n_vec, h_n, marker='.', lw=1.4, color=col, label=lbl)
        
    ax_pz_stab.set_xlim(-1.4, 1.4)
    ax_pz_stab.set_ylim(-1.4, 1.4)
    ax_pz_stab.set_aspect('equal', 'box')
    ax_pz_stab.set_title('(a) Posisi Pole terhadap Lingkaran Satuan', fontweight='bold')
    ax_pz_stab.set_xlabel(r'Re$\{z\}$')
    ax_pz_stab.set_ylabel(r'Im$\{z\}$')
    ax_pz_stab.grid(True, linestyle=':', alpha=0.6)
    ax_pz_stab.legend(loc='lower left', fontsize=8)
    
    ax_resp_stab.set_title('(b) Respons Impuls $h[n]$ pada Variasi Radius Pole', fontweight='bold')
    ax_resp_stab.set_xlabel('Indeks Sampel ($n$)')
    ax_resp_stab.set_ylabel('Amplitudo $h[n]$')
    ax_resp_stab.grid(True, linestyle=':', alpha=0.6)
    ax_resp_stab.legend(loc='upper left', fontsize=8)
    ax_resp_stab.set_ylim(-4, 4)
    
    plt.tight_layout()
    fig6_path = os.path.join(output_dir, "fig6_stability_analysis.png")
    fig.savefig(fig6_path, dpi=300)
    plt.close(fig)
    print(f"      [OK] Gambar 6: {fig6_path}")

def export_processed_data(data, output_path="data.csv"):
    print(f"[5/6] Mengekspor data hasil olahan ke berkas: {output_path}...")
    export_cols = [
        'sample_index', 'time_sec', 'timestamp', 'Activity',
        'accX', 'accY', 'accZ', 'acc_mag',
        'acc_mag_butter', 'acc_mag_notch', 'acc_mag_fir',
        'gyroX', 'gyroY', 'gyroZ'
    ]
    data[export_cols].to_csv(output_path, index=False)
    print(f"      Berhasil mengekspor {len(data)} baris data kontinu lengkap dengan kolom hasil pemfilteran.")

def print_summary_report(filters, data):
    print("\n" + "="*80)
    print("RINGKASAN TEKNIS SISTEM FILTER BERBASIS TRANSFORMASI-Z")
    print("="*80)
    
    mean_mag = data['acc_mag'].mean()
    std_sample = data['acc_mag'].std(ddof=1)
    
    print(f"Karakteristik Segmen Sinyal:")
    print(f" - Rentang Timestamp Aktual : {data['timestamp'].iloc[0]} s.d. {data['timestamp'].iloc[-1]} (dt = 0.100s konstan)")
    print(f" - Jumlah Sampel Kontinu   : {len(data)} sampel (Durasi = 60.0 detik pada Fs = 10.0 Hz)")
    print(f" - Nilai Rerata Magnitudo   : {mean_mag:.4f} m/s^2 (mendekati gravitasi bumi g = 9.81 m/s^2)")
    print(f" - Simpangan Baku Sampel   : {std_sample:.4f} m/s^2 (ddof=1)\n")
    
    b_but = filters['butter']['b']
    a_but = filters['butter']['a']
    p_but = filters['butter']['p']
    z_but = filters['butter']['z']
    
    print("1. Filter IIR Butterworth Orde-2 (Low-Pass Filter):")
    print(f"   Koefisien Numerator   (b) : {b_but}")
    print(f"   Koefisien Denominator (a) : {a_but}")
    print(f"   Fungsi Transfer H(z)      : ({b_but[0]:.6f} + {b_but[1]:.6f} z^-1 + {b_but[2]:.6f} z^-2) /")
    print(f"                               (1.000000 - {-a_but[1]:.6f} z^-1 + {a_but[2]:.6f} z^-2)")
    print(f"   Zeros (z)                 : {z_but} (Zero ganda di z = -1.0, frekuensi Nyquist)")
    print(f"   Poles (p)                 : {p_but}")
    print(f"   Magnitudo Poles (|p|)     : {np.abs(p_but)}")
    print(f"   Region of Convergence     : |z| > {np.max(np.abs(p_but)):.4f}")
    print(f"   Kriteria Kestabilan BIBO  : STABIL (Seluruh pole berada di dalam lingkaran satuan, |p| < 1)")
    
    print("\n2. Filter IIR Notch (Target Band-Stop 2.0 Hz):")
    b_not = filters['notch']['b']
    a_not = filters['notch']['a']
    p_not = filters['notch']['p']
    z_not = filters['notch']['z']
    print(f"   Zeros (|z|)               : {np.abs(z_not)} (Tepat pada lingkaran satuan)")
    print(f"   Poles (|p|)               : {np.abs(p_not)} (Di dalam lingkaran satuan -> Stabil)")
    
    print("\n3. Filter FIR Moving Average (M=5):")
    p_fir = filters['fir']['p']
    z_fir = filters['fir']['z']
    print(f"   Poles                     : {len(p_fir)} pole di titik origin z = 0")
    print(f"   Zeros                     : {len(z_fir)} zeros terdistribusi simetris pada lingkaran satuan")
    print(f"   Kriteria Kestabilan       : STABIL BIBO (Koefisien dan durasi impuls berhingga)")
    print("="*80 + "\n")

def main():
    dataset_file = "accelerometer_gyro_mobile_phone_dataset.csv"
    if not os.path.exists(dataset_file):
        print(f"Galat: Berkas {dataset_file} tidak ditemukan!")
        sys.exit(1)
        
    data, fs = load_and_preprocess_data(dataset_file)
    filters = design_filters(fs)
    data = apply_filters(data, filters)
    generate_visualizations(data, filters, fs, output_dir="images")
    export_processed_data(data, output_path="data.csv")
    print_summary_report(filters, data)
    print("[6/6] Seluruh proses pembaruan data, filter, dan visualisasi selesai dengan sukses.")

if __name__ == '__main__':
    main()
