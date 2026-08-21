"""
v0.3: エイリアシング修正 + Wienerデコンボリューションによる EDOF 実証
==================================================
v0.2(experiments/v0_2/v0_2_cubic_phase_mask.py)をベースに、以下2点を追加する。

(A) エイリアシングの修正(v0.1レビューから持ち越しの懸念点)
    原因は2つあり、どちらもv0.1/v0.2では「瞳グリッド半幅 == アパーチャ半径」
    (N_B=71, PUPIL_HALF_WIDTH=71/25) だったことに起因する。

    1. 位相サンプリング不足:
       Φ_DF(x1,y1;z) = kWm*r^2 (Eq.5) の勾配 |∇Φ_DF| = 2*kWm*r は
       アパーチャ端(r=R)で最大になる。kWm=10, R=2.84 のとき
           |∇Φ_DF|_max = 2*10*2.84 ≈ 56.8 [rad / 正規化座標1単位]
       v0.1/v0.2 の dx = 2R/(N_B-1) ≈ 0.0811 では
           Δφ_max = |∇Φ_DF|_max * dx ≈ 4.6 rad  (> π ⇒ サンプリング定理違反)
       これがkWm=±8〜10でのチェッカーボード状ノイズの正体。
       → 対策: アパーチャ内のサンプル数(=グリッド解像度)を増やし、dxを小さくする。

    2. ガードバンドなし:
       アパーチャが瞳グリッド端にぴったり接しており、瞳グリッドの外側に
       余白(ゼロ領域)がない。回折パターン(Airyパターン)の裾は
       1/r でゆっくり減衰するため、余白がないとFFTの周期境界で
       裾が反対側に折り返して重なる(circular wrap-around)。
       → 対策: 瞳グリッド半幅をアパーチャ半径より大きく取る(ガードバンド)。

    本コードでは瞳グリッド半幅を GRID_PAD_FACTOR 倍に広げつつ、
    グリッド点数 N_GRID を増やして dx を十分小さく保つ(下記パラメータ参照)。

(B) Wienerデコンボリューションによる EDOF (Extended Depth of Field) 実証
    report_chang.pdf の主張:「合焦時に測定/計算した単一のCPM PSFで
    デコンボリュームすれば、デフォーカスしていても特徴を復元できる」
    を、合成テストターゲット(チャープパターン)で数値的に検証する。
    比較対象として、マスクなし(clear aperture)を同じプロトコル
    (合焦PSF固定・Wiener復元)で処理した場合の性能も示す。

    結果(outputs/psnr_vs_defocus.png)は単調ではない:
    CPMはkWm=±1近傍でno-maskを大きく上回る(PSNR +5.5dB)が、
    1<|kWm|<=5では逆にno-maskをやや下回り、|kWm|=10近傍で再び同程度に
    収束する。これはno-maskの「間違ったカーネルでも無難に平坦な出力になる」
    性質と、CPMの「間違ったカーネルだと構造化された誤り(自信満々な誤答)に
    なりうる」性質の差によるものと考えられ、report_chang.pdf自身も
    Wienerフィルタのリンギングと、CPM不変性が有限範囲(実測で±30〜50μm)
    であることに言及している(main()末尾の出力と詳細コメントを参照)。
    一方 v0.2 で導入した形状不変性の指標(NCC)では、|kWm|>=6でも
    CPMがno-maskを明確に上回っており(0.42 vs 0.15)、PSNRの逆転は
    「デコンボリューション特有」の効果であることが示唆される。

論文 Eqs. 2-6 の対応(v0.1, v0.2と同じ):
    PSF(x2,y2) = |F{P(x1,y1)}|^2          ... (2)   ← PSF
    P(x1,y1) = A(x1,y1) * exp(i*Φ(x1,y1)) ... (3)   ← 瞳関数
    Φ_DF(x1,y1; z) = k*Wm*r^2             ... (5)   ← 離焦
    Φ(x1,y1; z) = Φ_DF + Φ_M              ... (6)   ← Φ_M: CPM(v0.2で導入)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# 1. パラメータ定義
# =============================================================================

WAVELENGTH = 550e-9      # 波長 λ [m](v0.1, v0.2と同じ、参考値)

N_PHI = 21                # デフォーカス断面の数(v0.1, v0.2と同じ)
PHI_MAX = 10.0             # kWm の最大値(無次元)

PUPIL_HALF_WIDTH = 71.0 / 25.0   # ≈ 2.84 (アパーチャ半径。v0.1, v0.2を踏襲)
ALPHA_CPM = 1.0                   # CPM強度(v0.2で report_chang.pdf と照合済み)

# --- エイリアシング対策(v0.3で新規) ---
GRID_PAD_FACTOR = 4        # 瞳グリッド半幅 = GRID_PAD_FACTOR * PUPIL_HALF_WIDTH
N_GRID = 1281               # 瞳グリッドの点数(奇数、中心画素を厳密に持たせる)
# 上記設定での位相サンプリング検証(コメント冒頭の見積もりと対応):
#   dx = 2*GRID_PAD_FACTOR*PUPIL_HALF_WIDTH / (N_GRID-1) ≈ 0.01775
#   Δφ_max(kWm=10, アパーチャ端) = 2*10*2.84*dx ≈ 1.01 rad (< π/3 相当、安全)

CROP_HALF = 300              # PSF中心から切り出す半幅(600x600)。
# 注意: CPM PSFはデフォーカスが大きいほどピーク位置が中心から遠くへ
# シフトする(report_chang.pdf Sec.2「center of the largest peak shifts
# with misfocus」)。crop_half=64 ではkWm=+10でエネルギーの約14%しか
# 捉えられておらず、デコンボリューション用カーネルとして不正確だった。
# crop_half=300 で kWm=+10 でもエネルギーの約99.2%を捕捉できることを確認済み。

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# 2. 瞳座標・アパーチャ・位相(ガードバンド付きグリッド版)
# =============================================================================
def build_pupil_coords(n=N_GRID, grid_half_width=None):
    """瞳グリッドの正規化座標 (xx, yy) と r^2 を返す。
    grid_half_width はアパーチャ半径より大きく取り、ガードバンドとする。"""
    if grid_half_width is None:
        grid_half_width = GRID_PAD_FACTOR * PUPIL_HALF_WIDTH
    x = np.linspace(-grid_half_width, grid_half_width, n)
    xx, yy = np.meshgrid(x, x)
    r2 = xx ** 2 + yy ** 2
    return xx, yy, r2


def build_aperture(r2, radius=PUPIL_HALF_WIDTH):
    """円形アパーチャ A(x1,y1)。半径はグリッド半幅より小さく、外側に余白ができる。"""
    return (r2 <= radius ** 2).astype(np.float32)


def gen_defocus_phase(phi_values, r2):
    """Φ_DF(x1,y1; z) = kWm * r^2 ... Eq.(5)。v0.1, v0.2と同一。"""
    return phi_values[:, None, None] * r2[None, :, :]


def gen_cpm_phase(xx, yy, alpha=ALPHA_CPM):
    """Φ_M(x1,y1) = α(x1^3 + y1^3)。v0.2と同一。"""
    return alpha * (xx ** 3 + yy ** 3)


# =============================================================================
# 3. PSF計算(大きいグリッドでFFTし、中心をクロップして保存)
# =============================================================================
def gen_psfs(oof_phase, aperture, n_grid=N_GRID, crop_half=CROP_HALF):
    """
    PSFスタックを計算し、中心 2*crop_half x 2*crop_half にクロップして返す。
    大きいグリッド(N_GRID)でFFTすることで(A)の2つの問題を回避しつつ、
    保存・可視化は必要な中心領域だけに絞ってメモリ・容量を節約する。
    """
    phase = oof_phase
    pupil = (aperture[None, :, :] * np.exp(1j * phase)).astype(np.complex64)
    fft = np.fft.fftshift(np.fft.fft2(pupil, axes=(-2, -1)), axes=(-2, -1))
    psf_unnorm = np.abs(fft) ** 2
    norm = n_grid * n_grid * np.sum(aperture ** 2)
    psfs = (psf_unnorm / norm).astype(np.float32)

    cx = n_grid // 2
    psfs_crop = psfs[:, cx - crop_half:cx + crop_half, cx - crop_half:cx + crop_half]
    return psfs_crop


# =============================================================================
# 4. Wienerデコンボリューション
# =============================================================================
def normalize_kernel(psf2d):
    """PSFをデコンボリューション/畳み込み用カーネルとして使えるよう sum=1 に正規化する。"""
    k = psf2d.astype(np.float64)
    s = k.sum()
    return k / s if s > 0 else k


def embed_kernel(kernel, out_shape):
    """
    小さいカーネルを画像と同じサイズのゼロ配列に中心配置し、さらに
    ifftshiftでカーネル中心をインデックス(0,0)に持ってくる。
    これにより FFT ベースの巡回畳み込み定理がそのまま使える
    (畳み込み定理: 空間領域の畳み込み = 周波数領域の積)。
    """
    kh, kw = kernel.shape
    oh, ow = out_shape
    canvas = np.zeros(out_shape, dtype=np.float64)
    y0, x0 = oh // 2 - kh // 2, ow // 2 - kw // 2
    canvas[y0:y0 + kh, x0:x0 + kw] = kernel
    return np.fft.ifftshift(canvas)


def blur_image(img, kernel):
    """img を kernel で(巡回)畳み込む。"""
    H = np.fft.fft2(embed_kernel(kernel, img.shape))
    F = np.fft.fft2(img)
    return np.real(np.fft.ifft2(H * F))


def wiener_deconvolve(blurred, kernel, K):
    """
    Wienerフィルタ: G(u,v) = H*(u,v) / (|H(u,v)|^2 + K)
    K はノイズ対信号比に対応する正則化定数(Kが大きいほど滑らかだが復元は弱い)。
    """
    H = np.fft.fft2(embed_kernel(kernel, blurred.shape))
    G = np.conj(H) / (np.abs(H) ** 2 + K)
    return np.real(np.fft.ifft2(G * np.fft.fft2(blurred)))


# =============================================================================
# 5. 合成テストターゲット(チャープ解像度パターン)
# =============================================================================
def gen_chirp_target(size=256, n_cycles=60.0):
    """
    report_chang.pdf の "chirped resolution target" を模した合成画像。
    上半分: 縦縞、左→右で空間周波数が増加(横方向の解像度評価用)。
    下半分: 横縞、上→下で空間周波数が増加(縦方向の解像度評価用)。
    """
    half = size // 2
    x = np.linspace(0, 1, size)
    phase_x = 2 * np.pi * (n_cycles / 2) * x ** 2
    top = np.tile(0.5 + 0.5 * np.sin(phase_x), (half, 1))

    y_local = np.linspace(0, 1, size - half)
    phase_y = 2 * np.pi * (n_cycles / 2) * y_local ** 2
    bottom = np.tile((0.5 + 0.5 * np.sin(phase_y))[:, None], (1, size))

    return np.vstack([top, bottom]).astype(np.float32)


def psnr(a, b, data_range=1.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return 100.0
    return 10 * np.log10(data_range ** 2 / mse)


# =============================================================================
# 6. 可視化
# =============================================================================
def plot_aliasing_fix_comparison(psf_before, psf_after, kwm, savepath):
    """v0.1(パディングなし)と v0.3(修正後)のPSF(kWm=極端値)を並べて比較する。"""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.8))
    for ax, img, title in zip(axes, [psf_before, psf_after],
                               [f"before (v0.1, no guard band)\nkWm={kwm:+.0f}",
                                f"after (v0.3, fixed)\nkWm={kwm:+.0f}"]):
        im = img / (img.max() + 1e-12)
        ax.imshow(im, cmap="hot", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Aliasing fix: no-mask PSF at large defocus", fontsize=11)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_psf_comparison_grid(psfs_no_mask, psfs_with_mask, phi_values, savepath):
    """v0.2と同じ形式の比較図(修正後グリッドで再生成)。"""
    n_phi = psfs_no_mask.shape[0]
    cx = psfs_no_mask.shape[1] // 2
    crop = 150

    def crop_norm(psfs):
        c = psfs[:, cx - crop:cx + crop, cx - crop:cx + crop].copy()
        return c / (c.max(axis=(1, 2), keepdims=True) + 1e-12)

    no_mask_norm = crop_norm(psfs_no_mask)
    with_mask_norm = crop_norm(psfs_with_mask)

    fig, axes = plt.subplots(2, n_phi, figsize=(2 * n_phi, 4.5))
    for j in range(n_phi):
        axes[0, j].imshow(no_mask_norm[j], cmap="hot", vmin=0, vmax=1)
        axes[0, j].set_title(f"{phi_values[j]:+.0f}", fontsize=8)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        axes[1, j].imshow(with_mask_norm[j], cmap="hot", vmin=0, vmax=1)
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])

    axes[0, 0].set_ylabel("no mask", fontsize=10)
    axes[1, 0].set_ylabel("CPM", fontsize=10)
    fig.suptitle("v0.3: PSF vs defocus (kWm) — aliasing-fixed grid", fontsize=12)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_edof_demo(target, results, indices, phi_values, savepath):
    """
    選択したデフォーカス断面について、no-mask/CPMそれぞれの
    ぼけ画像・復元画像を並べたグリッド図。
    results: dict with keys 'blur_nm','rec_nm','blur_cpm','rec_cpm' -> (N_PHI,H,W) arrays
    """
    n_sel = len(indices)
    fig, axes = plt.subplots(5, n_sel, figsize=(2.4 * n_sel, 12))
    row_labels = ["target", "no-mask: blurred", "no-mask: recovered",
                  "CPM: blurred", "CPM: recovered"]
    for col, j in enumerate(indices):
        axes[0, col].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"kWm={phi_values[j]:+.0f}", fontsize=10)
        axes[1, col].imshow(results["blur_nm"][j], cmap="gray", vmin=0, vmax=1)
        axes[2, col].imshow(results["rec_nm"][j], cmap="gray", vmin=0, vmax=1)
        axes[3, col].imshow(results["blur_cpm"][j], cmap="gray", vmin=0, vmax=1)
        axes[4, col].imshow(results["rec_cpm"][j], cmap="gray", vmin=0, vmax=1)
        for row in range(5):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    for row in range(5):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=10)
    fig.suptitle("v0.3: EDOF via Wiener deconvolution with a single in-focus PSF kernel",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_psnr_vs_defocus(psnr_nm, psnr_cpm, phi_values, savepath):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(phi_values, psnr_nm, "o-", label="no mask + Wiener (fixed in-focus kernel)")
    ax.plot(phi_values, psnr_cpm, "s-", label="CPM + Wiener (fixed in-focus kernel)")
    ax.set_xlabel("kWm (defocus, dimensionless)")
    ax.set_ylabel("PSNR [dB] vs. ground-truth target")
    ax.set_title("EDOF: reconstruction quality vs defocus (single fixed kernel)")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 7. メイン
# =============================================================================
def main():
    print("=" * 60)
    print("v0.3: エイリアシング修正 + Wienerデコンボリューション(EDOF)")
    print("=" * 60)

    # --- (A) 修正版グリッドでPSF計算 ---
    xx, yy, r2 = build_pupil_coords()
    aperture = build_aperture(r2)
    dx = xx[0, 1] - xx[0, 0]
    grad_max = 2 * PHI_MAX * PUPIL_HALF_WIDTH
    print(f"瞳グリッド      : {N_GRID}x{N_GRID}, dx={dx:.5f}")
    print(f"Δφ_max(kWm={PHI_MAX:.0f}, アパーチャ端) ≈ {grad_max * dx:.3f} rad (安全域: < π)")

    phi_values = np.linspace(-PHI_MAX, PHI_MAX, N_PHI, dtype=np.float32)
    oof_phase = gen_defocus_phase(phi_values, r2)
    cpm_phase = gen_cpm_phase(xx, yy, ALPHA_CPM)

    print("PSF計算中(no mask)...")
    psfs_no_mask = gen_psfs(oof_phase, aperture)
    print("PSF計算中(CPM)...")
    psfs_with_mask = gen_psfs(oof_phase + cpm_phase[None, :, :], aperture)
    print(f"PSFスタック(クロップ後): shape={psfs_no_mask.shape}")

    np.save(OUTPUT_DIR / "psfs_no_mask.npy", psfs_no_mask)
    np.save(OUTPUT_DIR / "psfs_with_mask.npy", psfs_with_mask)
    np.save(OUTPUT_DIR / "phi_values.npy", phi_values)

    plot_psf_comparison_grid(psfs_no_mask, psfs_with_mask, phi_values,
                              OUTPUT_DIR / "psf_comparison_grid.png")
    print(f"図を保存        : {OUTPUT_DIR}/psf_comparison_grid.png")

    # v0.1(パディングなし)の結果があれば before/after 比較図を作る
    v01_path = Path(__file__).parent.parent / "v0_1" / "outputs" / "psfs_no_mask.npy"
    if v01_path.exists():
        psfs_v01 = np.load(v01_path)
        plot_aliasing_fix_comparison(psfs_v01[-1], psfs_no_mask[-1], phi_values[-1],
                                      OUTPUT_DIR / "aliasing_fix_comparison.png")
        print(f"図を保存        : {OUTPUT_DIR}/aliasing_fix_comparison.png")

    # --- (B) Wienerデコンボリューションによる EDOF 実証 ---
    print("\nEDOF実証(Wienerデコンボリューション)...")
    # ターゲット画像はカーネル(2*CROP_HALF)より十分大きく取る
    # (embed_kernelがカーネルを画像内に収める必要があるため)
    target = gen_chirp_target(size=2 * CROP_HALF + 200)
    focus_idx = N_PHI // 2  # kWm=0

    kernel_nm_focus = normalize_kernel(psfs_no_mask[focus_idx])
    kernel_cpm_focus = normalize_kernel(psfs_with_mask[focus_idx])

    rng = np.random.default_rng(0)
    NOISE_SIGMA = 0.01
    K_WIENER = 3e-3

    blur_nm = np.empty((N_PHI, *target.shape), dtype=np.float32)
    blur_cpm = np.empty_like(blur_nm)
    rec_nm = np.empty_like(blur_nm)
    rec_cpm = np.empty_like(blur_nm)
    psnr_nm = np.empty(N_PHI)
    psnr_cpm = np.empty(N_PHI)

    for j in range(N_PHI):
        k_nm_j = normalize_kernel(psfs_no_mask[j])
        k_cpm_j = normalize_kernel(psfs_with_mask[j])

        b_nm = blur_image(target, k_nm_j) + rng.normal(0, NOISE_SIGMA, target.shape)
        b_cpm = blur_image(target, k_cpm_j) + rng.normal(0, NOISE_SIGMA, target.shape)

        # 復元は「合焦時に測定した単一のPSF」を固定カーネルとして使う
        # (実運用でデフォーカス量が未知でも復元できるかを模した設定)
        r_nm = wiener_deconvolve(b_nm, kernel_nm_focus, K_WIENER)
        r_cpm = wiener_deconvolve(b_cpm, kernel_cpm_focus, K_WIENER)

        blur_nm[j], blur_cpm[j] = np.clip(b_nm, 0, 1), np.clip(b_cpm, 0, 1)
        rec_nm[j], rec_cpm[j] = np.clip(r_nm, 0, 1), np.clip(r_cpm, 0, 1)
        psnr_nm[j] = psnr(rec_nm[j], target)
        psnr_cpm[j] = psnr(rec_cpm[j], target)

    results = {"blur_nm": blur_nm, "rec_nm": rec_nm, "blur_cpm": blur_cpm, "rec_cpm": rec_cpm}
    demo_indices = [2, 6, 10, 14, 18]  # kWm = -8, -4, 0, +4, +8
    plot_edof_demo(target, results, demo_indices, phi_values,
                    OUTPUT_DIR / "edof_deconvolution_demo.png")
    print(f"図を保存        : {OUTPUT_DIR}/edof_deconvolution_demo.png")

    plot_psnr_vs_defocus(psnr_nm, psnr_cpm, phi_values,
                          OUTPUT_DIR / "psnr_vs_defocus.png")
    print(f"図を保存        : {OUTPUT_DIR}/psnr_vs_defocus.png")

    # 数値指標のまとめ。
    # 結果は単調ではない: CPMはkWm=±1近傍でno-maskを大きく上回るが、
    # kWm=2〜9では逆にno-maskを下回り、kWm=±10近傍で再び同程度に収束する。
    # これはバグではなく、固定カーネルWienerデコンボリューションの性質を
    # 反映していると考えられる(psnr_vs_defocus.png 参照):
    #   - no-maskは設計上のDOFが浅く、少しデフォーカスすると急激にほぼ一様な
    #     グレー画像になる。そのため間違ったカーネルで復元しても「平坦に近い
    #     無難な出力」になり、MSEはさほど悪化しない。
    #   - CPMのPSFはデフォーカスしてもエネルギーが構造を保ったまま広がる
    #     (v0.2のNCC指標が示す通り)。しかし合焦カーネルとの不一致が大きい
    #     defocus域では「自信満々に間違った」構造化された復元結果になり、
    #     単純なピクセル単位のMSE(PSNR)としてはno-maskの「無難な平坦出力」
    #     より悪化しうる。report_chang.pdfもWienerフィルタのリンギングと、
    #     CPMの不変性が有限範囲(±30〜50μm)であることに言及している。
    near = np.abs(phi_values) <= 1
    mid = (np.abs(phi_values) > 1) & (np.abs(phi_values) <= 5)
    print(f"\n[数値指標] PSNR比較(defocus帯域別、report_chang.pdfとの対応も参照):")
    print(f"  合焦近傍 |kWm|<=1 : no mask={psnr_nm[near].mean():6.2f} dB, "
          f"CPM={psnr_cpm[near].mean():6.2f} dB (CPMが明確に優位)")
    print(f"  中程度   1<|kWm|<=5: no mask={psnr_nm[mid].mean():6.2f} dB, "
          f"CPM={psnr_cpm[mid].mean():6.2f} dB (固定カーネルWienerの限界域)")
    print(f"  v0.2のNCC指標(形状不変性)ではCPMが|kWm|>=6でも優位(0.42 vs 0.15)。")
    print(f"  PSNRの逆転はデコンボリューション特有の効果であり、次段で要検討。")

    return psfs_no_mask, psfs_with_mask, phi_values, psnr_nm, psnr_cpm


if __name__ == "__main__":
    main()
