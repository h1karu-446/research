"""
v0.2b: 立方位相マスク(CPM)PSFシミュレーション — エイリアシング修正版
==================================================
v0_2_cubic_phase_mask.py のロジック(CPM位相・PSF比較図・z不変性のNCC指標)は
そのままに、v0.1レビューから持ち越しだったエイリアシングの懸念点を解消する。

原因は2つあり、どちらも「瞳グリッド半幅 == アパーチャ半径」
(N_B=71, PUPIL_HALF_WIDTH=71/25) だったことに起因する。

1. 位相サンプリング不足:
   Φ_DF(x1,y1;z) = kWm*r^2 (デフォーカス位相) の勾配 |∇Φ_DF| = 2*kWm*r は
   アパーチャ端(r=R)で最大になる。kWm=10, R=2.84 のとき
       |∇Φ_DF|_max = 2*10*2.84 ≈ 56.8 [rad / 正規化座標1単位]
   旧版の dx = 2R/(N_B-1) ≈ 0.0811 (隣り合う点と点の距離)では
       Δφ_max = |∇Φ_DF|_max * dx ≈ 4.6 rad  (> π ⇒ サンプリング定理違反, 本来の位相変化が適切に反映されない。270°時計回り回転は、90度の反時計回り回転として捉えられてしまう)
   これがkWm=±8〜10でのチェッカーボード状ノイズの正体。
   → 対策: アパーチャ内のサンプル数(=グリッド解像度)を増やし、dxを小さくする。

2. ガードバンドなし:
   アパーチャが瞳グリッド端にぴったり接しており、瞳グリッドの外側に
   余白(ゼロ領域)がない。回折パターン(Airyパターン)の裾は
   1/r でゆっくり減衰するため、余白がないとFFTの周期境界で
   裾が反対側に折り返して重なる(circular wrap-around)。
   → 対策: 瞳グリッド半幅をアパーチャ半径より大きく取る(ガードバンド)。

本コードでは瞳グリッド半幅を GRID_PAD_FACTOR 倍に広げつつ、グリッド点数
N_GRID を増やして dx を十分小さく保つ(下記パラメータ参照)。
CPM PSFはデフォーカスが大きいほどピーク位置が中心から遠くへシフトする
ため、PSFの保存・可視化用クロップ幅(CROP_HALF)も併せて拡張している。

論文 Eqs. 2-6 の対応(v0.1, v0.2と同じ):
    PSF(x2,y2) = |F{P(x1,y1)}|^2          ... (2)   ← PSF
    P(x1,y1) = A(x1,y1) * exp(i*Φ(x1,y1)) ... (3)   ← 瞳関数
    Φ_DF(x1,y1; z) = k*Wm*r^2             ... (5)   ← 離焦
    Φ(x1,y1; z) = Φ_DF + Φ_M              ... (6)   ← Φ_M: CPM
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

# --- CPM パラメータ ---
# 中心→瞳コーナーでの位相量 α(x³+y³): α=1.0 のとき Δφ ≈ 45.8 rad ≈ 7.3 cycle。
# report_chang.pdf Sec.3の基準(対角線上で7-8 cycle)にほぼ一致(詳細はv0_2参照)。
ALPHA_CPM = 1.0

# --- エイリアシング対策 ---
GRID_PAD_FACTOR = 4        # 瞳グリッド全体の広さを、アパーチャ(実際のレンズの開口)の半径の何倍にするかを決める倍率。瞳グリッド半幅 = GRID_PAD_FACTOR * PUPIL_HALF_WIDTH
N_GRID = 1281               # 瞳グリッドの点数(奇数、中心画素を厳密に持たせる)。グリッド数を増やしてサンプリング不足を解消
# 上記設定での位相サンプリング検証:
#   dx = 2*GRID_PAD_FACTOR*PUPIL_HALF_WIDTH / (N_GRID-1) ≈ 0.01775
#   Δφ_max(kWm=10, アパーチャ端) = 2*10*2.84*dx ≈ 1.01 rad (< π/3 相当、安全)

CROP_HALF = 300              # PSF中心から切り出す半幅(600x600)。
# GRID_PAD_FACTOR(4倍)の大きさのキャンパスを用意した(ガードバンドのため)が、実際にPSFで意味をなすのはもっと狭い範囲だけ。
# crop_half=64(旧v0.2相当)ではkWm=+10でCPM PSFのエネルギーの約14%しか
# 捉えられない。crop_half=300でkWm=+10でも約99.2%を捕捉できることを確認済み。

OUTPUT_DIR = Path(__file__).parent / "outputs_aliasing_fixed"  # 旧v0.2の outputs/ とは別フォルダに分離
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# 2. 瞳座標・アパーチャ・位相(ガードバンド付きグリッド版)
# =============================================================================
def build_pupil_coords(n=N_GRID, grid_half_width=None):
    """瞳グリッドの正規化座標 (xx, yy) と r^2 を返す。
    grid_half_width はアパーチャ半径より大きく取り、ガードバンドとする。"""
    if grid_half_width is None:
        grid_half_width = GRID_PAD_FACTOR * PUPIL_HALF_WIDTH  # グリッド全体の広さを計算
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
    大きいグリッド(N_GRID)でFFTすることでエイリアシングの2つの原因を回避しつつ、
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
# 4. 数値指標:PSFのz不変性を定量化する(v0.2と同一ロジック)
# =============================================================================
def compute_invariance_metric(psfs, crop=150, ref_idx=None):
    """
    各デフォーカス断面のPSFと合焦断面(既定でref_idx=中央、kWm=0)のPSFとの
    正規化相互相関係数(NCC)を計算する。NCC=1に近いほどz不変性が高い。

    report_chang.pdf (Sec.2) が述べる通り、CPMのPSFは「形状はほぼ同じだが
    ピーク位置がデフォーカスに応じて横シフトする」。固定窓での相関だと
    このシフトが形状変化と誤認されるため、各断面をピーク位置basisで
    クロップ(位置合わせ)してから比較する(v0.2と同じロジック。
    crop のデフォルト値だけ、PSFの保存サイズが128→600になったのに
    合わせて 16→150 にスケールしてある)。
    """
    n_phi, n, _ = psfs.shape
    if ref_idx is None:
        ref_idx = n_phi // 2

    padded = np.pad(psfs, ((0, 0), (crop, crop), (crop, crop)), mode="constant")

    def peak_crop(j):
        img = padded[j]
        py, px = np.unravel_index(np.argmax(img), img.shape)
        return img[py - crop:py + crop, px - crop:px + crop]

    ref = peak_crop(ref_idx).astype(np.float64)
    ref = ref - ref.mean()
    ref_norm = np.linalg.norm(ref)

    ncc = np.empty(n_phi)
    for j in range(n_phi):
        cur = peak_crop(j).astype(np.float64)
        cur = cur - cur.mean()
        cur_norm = np.linalg.norm(cur)
        ncc[j] = np.dot(ref.ravel(), cur.ravel()) / (ref_norm * cur_norm + 1e-12)
    return ncc


def plot_invariance_comparison(ncc_no_mask, ncc_with_mask, phi_values, savepath):
    """マスクなし/ありのNCC(z不変性指標)をkWmに対してプロットする。"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(phi_values, ncc_no_mask, "o-", label="no mask")
    ax.plot(phi_values, ncc_with_mask, "s-", label="CPM")
    ax.set_xlabel("kWm (defocus, dimensionless)")
    ax.set_ylabel("NCC vs in-focus PSF")
    ax.set_title("PSF shape invariance across defocus (higher = more z-invariant)")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 5. 可視化
# =============================================================================
def plot_psf_comparison_grid(psfs_no_mask, psfs_with_mask, phi_values, savepath):
    """
    マスクなし/ありのPSFを2段に並べて比較する図(v0.2と同形式、修正後グリッド版)。
    """
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
    fig.suptitle("v0.2b: PSF vs defocus (kWm) — aliasing-fixed grid", fontsize=12)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_aliasing_fix_comparison(psf_before, psf_after, kwm, savepath):
    """v0.1(パディングなし)と本ファイル(修正後)のPSF(kWm=極端値)を並べて比較する。"""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.8))
    for ax, img, title in zip(axes, [psf_before, psf_after],
                               [f"before (v0.1, no guard band)\nkWm={kwm:+.0f}",
                                f"after (v0.2b, fixed)\nkWm={kwm:+.0f}"]):
        im = img / (img.max() + 1e-12)
        ax.imshow(im, cmap="hot", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Aliasing fix: no-mask PSF at large defocus", fontsize=11)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 6. メイン
# =============================================================================
def main():
    print("=" * 60)
    print("v0.2b: 立方位相マスク(CPM)PSFシミュレーション — エイリアシング修正版")
    print("=" * 60)

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

    # 保存(旧v0.2の出力とは outputs_aliasing_fixed/ で分離しているので、ファイル名は素のまま)
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

    # 数値指標:z不変性(NCC)の計算と可視化
    ncc_no_mask = compute_invariance_metric(psfs_no_mask)
    ncc_with_mask = compute_invariance_metric(psfs_with_mask)
    plot_invariance_comparison(ncc_no_mask, ncc_with_mask, phi_values,
                                OUTPUT_DIR / "invariance_metric.png")
    print(f"図を保存        : {OUTPUT_DIR}/invariance_metric.png")
    print(f"\n[数値指標] 合焦PSFとのNCC平均(|kWm|>=6の範囲, z不変性の指標):")
    far = np.abs(phi_values) >= 6
    print(f"  no mask : {ncc_no_mask[far].mean():.4f}")
    print(f"  CPM     : {ncc_with_mask[far].mean():.4f}")
    print(f"  (旧v0.2でのエイリアシングありの値: no mask=0.1483, CPM=0.4208)")

    return psfs_no_mask, psfs_with_mask, phi_values, ncc_no_mask, ncc_with_mask


if __name__ == "__main__":
    main()
