"""
v0.1: 標準PSF（マスクなし）シミュレーション
==================================================
DeepDOF (Jin et al. 2020) リポジトリの DeepDOF_step2.py を参考に、
位相マスクを「外した」状態の通常の顕微鏡PSFをシミュレートする。

論文 Eqs. 2–6 の対応:
    PSF(x2,y2) = |F{P(x1,y1)}|^2          ... (2)
    P(x1,y1) = A(x1,y1) * exp(i*Φ(x1,y1)) ... (3)
    Φ_DF(x1,y1; z) = k*Wm*r^2             ... (5)   ← 離焦のみ
    Φ(x1,y1; z) = Φ_DF + Φ_M              ... (6)   ← マスクなし = Φ_M = 0

DeepDOF_step2.py の gen_OOFphase / gen_PSFs を NumPy に書き直したもの。
TensorFlow 1.x の自動微分は v0.1 では不要なので外している。

参考: 論文では kWm を [-10, +10] の 21 点で離散化し、これが
物理デフォーカス ±100 μm（200 μm DOFレンジ）に対応する。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# 1. パラメータ定義
# =============================================================================

# --- 物理パラメータ（DeepDOF と同じ：Olympus 4x/0.13 NA, λ=550 nm）---
WAVELENGTH = 550e-9      # 波長 λ [m]（緑、蛍光イメージング）
N_REFRACT  = 1.5         # マスク基板の屈折率（v0.1 では未使用、v0.2 で使う）

# --- シミュレーションパラメータ ---
N_B    = 71              # PSF と瞳の格子サイズ [pix]（DeepDOFと同じ）
N_PHI  = 21              # デフォーカス断面の数（DeepDOFと同じ）
PHI_MAX = 10.0           # kWm の最大値（無次元）→ 物理的 ±100 μm に相当

# 瞳座標の正規化半径
# DeepDOF の x0 = linspace(-2.84, 2.84, 71) は r=71/25 に由来する。
# これは「PSF と瞳のサンプリング条件（センサピクセル 0.72 μm）」を
# 決め打ちした値。我々も同じにしておけば論文と直接比較できる。
PUPIL_HALF_WIDTH = 71.0 / 25.0   # ≈ 2.84

# 出力ディレクトリ
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# 2. 瞳座標と瞳マスク（円形アパーチャ）
# =============================================================================
def build_pupil_coords(n=N_B, half_width=PUPIL_HALF_WIDTH):
    """瞳面の正規化座標 (xx, yy) と r^2 を返す。"""
    x = np.linspace(-half_width, half_width, n)
    xx, yy = np.meshgrid(x, x)
    r2 = xx ** 2 + yy ** 2
    return xx, yy, r2


def build_aperture(r2, radius=PUPIL_HALF_WIDTH):
    """円形アパーチャ A(x1,y1) を返す。瞳の内側で 1、外側で 0。

    DeepDOF のコードでは zernike_basis_150mm.mat に格納された idx を
    使っているが、それは「瞳半径 = 格子の最大値」と同じ円形アパーチャ。
    我々はシンプルに自前で作る。
    """
    return (r2 <= radius ** 2).astype(np.float32)


# =============================================================================
# 3. デフォーカス位相 Φ_DF（論文 Eq. 5）
# =============================================================================
def gen_defocus_phase(phi_values, r2):
    """各デフォーカス値に対する Φ_DF(x1,y1; z) = kWm * r^2 を生成する。

    Parameters
    ----------
    phi_values : (N_Phi,) ndarray
        各断面の kWm 値（無次元）。DeepDOF では linspace(-10, 10, 21)。
    r2 : (N, N) ndarray
        瞳面の r^2。

    Returns
    -------
    oof_phase : (N_Phi, N, N) ndarray
        デフォーカス位相 [rad]。
    """
    # broadcasting: (N_Phi, 1, 1) * (1, N, N) → (N_Phi, N, N)
    return phi_values[:, None, None] * r2[None, :, :]


# =============================================================================
# 4. PSF 計算（マスクなし版、論文 Eq. 2–6 の Φ_M = 0 ケース）
# =============================================================================
def gen_psfs_no_mask(oof_phase, aperture):
    """マスクなしの PSF スタックを生成する。

    DeepDOF_step2.py の gen_PSFs を NumPy に移植。
    マスクの高さマップ h=0 とすることで Φ_M=0 を実現している。

    Parameters
    ----------
    oof_phase : (N_Phi, N, N) ndarray
        各断面のデフォーカス位相 [rad]。
    aperture : (N, N) ndarray
        円形アパーチャ A(x1,y1)。

    Returns
    -------
    psfs : (N_Phi, N, N) ndarray
        各デフォーカスでの PSF（正規化済）。
    """
    n_phi, n, _ = oof_phase.shape

    # 瞳関数 P = A * exp(i*Φ)。Φ_M=0 なので Φ = Φ_DF のみ。
    phase = oof_phase                           # (N_Phi, N, N)
    pupil = aperture[None, :, :] * np.exp(1j * phase)

    # FFT → fftshift → 強度の二乗 → 規格化
    # DeepDOF と同じく Norm = N*N*sum(A^2)
    fft = np.fft.fftshift(np.fft.fft2(pupil), axes=(-2, -1))
    psf_unnorm = np.abs(fft) ** 2
    norm = n * n * np.sum(aperture ** 2)
    psfs = psf_unnorm / norm
    return psfs.astype(np.float32)


# =============================================================================
# 5. 可視化
# =============================================================================
def plot_psf_grid_normalized(psfs, phi_values, savepath):
    """各PSFを個別正規化してグリッド表示。
    
    ピーク強度の絶対値ではなく、PSF形状そのものの変化（焦点でシャープ→
    離焦で広がる）を見るための図。
    """
    n_phi = psfs.shape[0]
    cx = psfs.shape[1] // 2
    crop = 16
    psfs_crop = psfs[:, cx - crop:cx + crop, cx - crop:cx + crop].copy()

    # 各PSFを個別に max=1 で正規化
    psfs_norm = psfs_crop / (psfs_crop.max(axis=(1, 2), keepdims=True) + 1e-12)

    fig, axes = plt.subplots(3, 7, figsize=(14, 6.5))
    for j, ax in enumerate(axes.flat):
        if j >= n_phi:
            ax.axis("off")
            continue
        ax.imshow(psfs_norm[j], cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"kWm={phi_values[j]:+.0f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("v0.1: Standard PSF (no mask) - each PSF normalized to its own peak", fontsize=12)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_psf_grid(psfs, phi_values, savepath):
    """21枚のPSFをグリッド表示。中心切り出しで詳細を見やすくする。"""
    n_phi = psfs.shape[0]
    # PSFの中心32x32だけ表示（71x71全部はスカスカで見にくい）
    cx = psfs.shape[1] // 2
    crop = 16
    psfs_crop = psfs[:, cx - crop:cx + crop, cx - crop:cx + crop]

    # 共通カラースケール（焦点位置で正規化）
    vmax = psfs_crop.max()

    fig, axes = plt.subplots(3, 7, figsize=(14, 6.5))
    for j, ax in enumerate(axes.flat):
        if j >= n_phi:
            ax.axis("off")
            continue
        ax.imshow(psfs_crop[j], cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(f"kWm={phi_values[j]:+.0f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("v0.1: Standard PSF (no mask) - PSF at 21 defocus planes", fontsize=12)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_psf_profiles(psfs, phi_values, savepath):
    """PSF中心横断面（x方向プロファイル）を5断面だけ重ねて描く。"""
    cx = psfs.shape[1] // 2
    pick = [0, 5, 10, 15, 20]  # -10, -5, 0, +5, +10
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j in pick:
        prof = psfs[j, cx, :]
        ax.plot(prof, label=f"kWm={phi_values[j]:+.0f}")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("PSF intensity")
    ax.set_title("PSF cross-section profile (center row)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_peak_vs_defocus(psfs, phi_values, savepath):
    """ピーク強度 vs デフォーカス。"""
    peaks = psfs.max(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(phi_values, peaks, "o-")
    ax.set_xlabel("kWm (defocus, dimensionless)")
    ax.set_ylabel("PSF peak intensity")
    ax.set_title("PSF peak vs defocus (proxy for DOF)")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 6. メイン
# =============================================================================
def main():
    print("=" * 60)
    print("v0.1: 標準PSFシミュレーション（マスクなし）")
    print("=" * 60)

    # 座標とアパーチャ
    xx, yy, r2 = build_pupil_coords()
    aperture = build_aperture(r2)
    print(f"格子サイズ      : {N_B}x{N_B}")
    print(f"アパーチャ面積  : {int(aperture.sum())} pixels (円形)")

    # デフォーカス値（DeepDOFと同じ）
    phi_values = np.linspace(-PHI_MAX, PHI_MAX, N_PHI, dtype=np.float32)
    print(f"デフォーカス断面: {N_PHI}枚, kWm ∈ [{phi_values[0]:+.0f}, {phi_values[-1]:+.0f}]")

    # 計算
    oof_phase = gen_defocus_phase(phi_values, r2)
    psfs = gen_psfs_no_mask(oof_phase, aperture)
    print(f"PSFスタック     : shape={psfs.shape}, dtype={psfs.dtype}")
    print(f"焦点(kWm=0) PSF : peak={psfs[10].max():.4e}, sum={psfs[10].sum():.4f}")
    print(f"離焦(kWm=10)PSF : peak={psfs[20].max():.4e}, sum={psfs[20].sum():.4f}")

    # 保存
    np.save(OUTPUT_DIR / "psfs_no_mask.npy", psfs)
    np.save(OUTPUT_DIR / "phi_values.npy", phi_values)
    print(f"\nNumPy配列を保存 : {OUTPUT_DIR / 'psfs_no_mask.npy'}")

    # 可視化
    plot_psf_grid(psfs, phi_values, OUTPUT_DIR / "psf_grid.png")
    plot_psf_grid_normalized(psfs, phi_values, OUTPUT_DIR / "psf_grid_normalized.png")
    plot_psf_profiles(psfs, phi_values, OUTPUT_DIR / "psf_profiles.png")
    plot_peak_vs_defocus(psfs, phi_values, OUTPUT_DIR / "peak_vs_defocus.png")
    print(f"図を保存        : {OUTPUT_DIR}/*.png")

    return psfs, phi_values


if __name__ == "__main__":
    psfs, phi_values = main()
