"""
v0.2: 立方位相マスク(CPM)PSFシミュレーション
==================================================
v0.1(experiments/v0_1/v0_1_standard_psf.py)の gen_psfs_no_mask をベースに、
瞳の位相へ Cubic Phase Mask Φ_M(x1,y1) = a(x1^3 + y1^3) を追加する。

論文 Eqs. 2-6 の対応(v0.1と同じ):
    PSF(x2,y2) = |F{P(x1,y1)}|^2          ... (2)   ← PSF
    P(x1,y1) = A(x1,y1) * exp(i*Φ(x1,y1)) ... (3)   ← 瞳関数
    Φ_DF(x1,y1; z) = k*Wm*r^2             ... (5)   ← 離焦のみ
    Φ(x1,y1; z) = Φ_DF + Φ_M              ... (6)   ← v0.1では Φ_M = 0

v0.2では Φ_M = a(x1^3 + y1^3) (report_chang.pdf Sec.2, Dowski & Cathey 1995) を
Eq.(6) に足すことで、PSFのz方向不変性が現れるかを検証する。

report_chang.pdf との照合結果:
    - 同論文Fig.2の "cpm psf" は格子状の輝点が片隅に偏在する非対称パターン。
      本コードの psf_comparison_grid.png のCPM行も同様の構造を示しており、
      定性的に一致することを確認済み。
    - 同論文Sec.3: 「瞳の対角線上で7-8位相サイクル入るようαを選ぶ」。
      本コードの ALPHA_CPM=1.0 は中心→瞳コーナーで約7.3 cycleとなり、
      この基準にほぼ一致(詳細は ALPHA_CPM の定義部コメント参照)。
    - z不変性の数値指標(NCC, ピーク位置合わせ済み)を追加し、
      outputs/invariance_metric.png に保存。|kWm|>=6 の平均NCCで
      CPM(約0.42) が no mask(約0.15) を明確に上回ることを確認。

残る懸念点(v0.1レビューより持ち越し、v0.3着手前に対応を検討):
    1. アパーチャがグリッド端に接しておりゼロパディングがない(エイリアシングの疑い)。
       |kWm|>=8 付近の comparison_grid.png にチェッカーボード状ノイズとして表れている。
    2. PUPIL_HALF_WIDTH の物理的根拠が未導出(v0.1〜v0.3は踏襲、v0.4で要導出)。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# 1. パラメータ定義
# =============================================================================

# --- 物理パラメータ(v0.1と同じ:Olympus 4x/0.13 NA, λ=550 nm) ---
WAVELENGTH = 550e-9      # 波長 λ [m]

# --- シミュレーションパラメータ(v0.1と同じ) ---
N_B    = 71              # PSF と瞳の格子サイズ [pix]
N_PHI  = 21               # デフォーカス断面の数
PHI_MAX = 10.0            # kWm の最大値(無次元)

PUPIL_HALF_WIDTH = 71.0 / 25.0   # ≈ 2.84（v0.1を踏襲。根拠はv0.4で要導出）

# --- CPM パラメータ(v0.2で新規追加) ---
# report_chang.pdf (Sec.3): 「対角線上で7-8位相サイクル入るようにαを選ぶ」
# (原論文は α=8π をmm単位の座標で使用。単位系が違うので数値はそのまま流用不可)。
# 中心→瞳コーナーでの位相量 α(x³+y³) を確認: α=1.0 のとき
#   Δφ = 1.0 * 2*(2.84)^3 ≈ 45.8 rad ≈ 7.3 cycles
# となり、論文の基準(7-8 cycles)にほぼ一致することを確認済み。
ALPHA_CPM = 1.0

# 出力ディレクトリ
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# 2. 瞳座標と瞳マスク(円形アパーチャ)— v0.1と同じ
# =============================================================================
def build_pupil_coords(n=N_B, half_width=PUPIL_HALF_WIDTH):
    """瞳面の正規化座標 (xx, yy) と r^2 を返す。"""
    x = np.linspace(-half_width, half_width, n)
    xx, yy = np.meshgrid(x, x)
    r2 = xx ** 2 + yy ** 2
    return xx, yy, r2


def build_aperture(r2, radius=PUPIL_HALF_WIDTH):
    """円形アパーチャ A(x1,y1) を返す。瞳の内側で1、外側で0。"""
    return (r2 <= radius ** 2).astype(np.float32)


# =============================================================================
# 3. 位相生成:デフォーカス(v0.1と同じ)+ CPM(v0.2で新規、ここが穴)
# =============================================================================
def gen_defocus_phase(phi_values, r2):
    """
    各デフォーカス値に対する Φ_DF(x1,y1; z) = kWm * r^2 を生成する。
    (N_Phi, N, N) の3次元配列を返す。v0.1と同一。
    """
    return phi_values[:, None, None] * r2[None, :, :]


def gen_cpm_phase(xx, yy, alpha=ALPHA_CPM):
    """
    立方位相マスク Φ_M(x1,y1) = α(x1^3 + y1^3) を生成する。

    xx, yy: build_pupil_coords() が返す瞳面の正規化座標 (N, N)
    alpha : マスクの強さ ALPHA_CPM

    戻り値: Φ_M の位相マップ (N, N)。デフォーカスに依らず一定
           (これが「PSFのz不変性」を生む肝)。
    """
    return alpha * (xx ** 3 + yy ** 3)


# =============================================================================
# 4. PSF計算
# =============================================================================
def gen_psfs_no_mask(oof_phase, aperture):
    """マスクなしのPSFスタックを生成する(v0.1と同一、比較用ベースライン)。"""
    n_phi, n, _ = oof_phase.shape
    phase = oof_phase
    pupil = aperture[None, :, :] * np.exp(1j * phase)
    fft = np.fft.fftshift(np.fft.fft2(pupil), axes=(-2, -1))
    psf_unnorm = np.abs(fft) ** 2
    norm = n * n * np.sum(aperture ** 2)
    psfs = psf_unnorm / norm
    return psfs.astype(np.float32)


def gen_psfs_with_mask(oof_phase, cpm_phase, aperture):
    """
    CPMありのPSFスタックを生成する。
    Φ(x1,y1;z) = Φ_DF(x1,y1;z) + Φ_M(x1,y1)  ... Eq.(6)
    cpm_phase は z に依らず全断面で共通なのでブロードキャストで足す。
    """
    n_phi, n, _ = oof_phase.shape
    phase = oof_phase + cpm_phase[None, :, :]   # Eq.(6)
    pupil = aperture[None, :, :] * np.exp(1j * phase)
    fft = np.fft.fftshift(np.fft.fft2(pupil), axes=(-2, -1))
    psf_unnorm = np.abs(fft) ** 2
    norm = n * n * np.sum(aperture ** 2)
    psfs = psf_unnorm / norm
    return psfs.astype(np.float32)


# =============================================================================
# 5. 数値指標:PSFのz不変性を定量化する
# =============================================================================
def compute_invariance_metric(psfs, crop=16, ref_idx=None):
    """
    各デフォーカス断面のPSFと合焦断面(既定でref_idx=中央、kWm=0)のPSFとの
    正規化相互相関係数(NCC)を計算する。
    NCC=1に近いほど「形状が合焦時と変わらない」= z不変性が高いことを意味する。

    report_chang.pdf (Sec.2) が述べる通り、CPMのPSFは「形状はほぼ同じだが
    ピーク位置がデフォーカスに応じて横シフトする」。固定窓での相関だと
    このシフトが形状変化と誤認され、z不変性を過小評価してしまう。
    そのため各断面をピーク位置basisでクロップ(位置合わせ)してから
    比較する(平行移動を無視し、純粋な形状の一致度を見る)。
    """
    n_phi, n, _ = psfs.shape
    if ref_idx is None:
        ref_idx = n_phi // 2

    # ピークが端に近くてもクロップ窓が切れないよう、あらかじめゼロパディングする
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
# 6. 可視化
# =============================================================================
def plot_psf_comparison_grid(psfs_no_mask, psfs_with_mask, phi_values, savepath):
    """
    マスクなし/ありのPSFを2段に並べて比較する図。
    report_chang.pdf のFig.に類似する挙動(マスクありがz方向にほぼ不変)
    が出るかを見るための図。各PSFは個別にmax=1正規化。
    """
    n_phi = psfs_no_mask.shape[0]
    cx = psfs_no_mask.shape[1] // 2
    crop = 16

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
    fig.suptitle("v0.2: PSF vs defocus (kWm) — no mask (top) vs CPM (bottom)", fontsize=12)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 7. メイン
# =============================================================================
def main():
    print("=" * 60)
    print("v0.2: 立方位相マスク(CPM)PSFシミュレーション")
    print("=" * 60)

    # 座標とアパーチャ
    xx, yy, r2 = build_pupil_coords()
    aperture = build_aperture(r2)
    print(f"格子サイズ      : {N_B}x{N_B}")
    print(f"アパーチャ面積  : {int(aperture.sum())} pixels (円形)")

    # デフォーカス値(v0.1と同じ)
    phi_values = np.linspace(-PHI_MAX, PHI_MAX, N_PHI, dtype=np.float32)
    print(f"デフォーカス断面: {N_PHI}枚, kWm ∈ [{phi_values[0]:+.0f}, {phi_values[-1]:+.0f}]")

    # 位相計算
    oof_phase = gen_defocus_phase(phi_values, r2)
    cpm_phase = gen_cpm_phase(xx, yy, ALPHA_CPM)  

    # PSF計算(マスクなし/ありの両方を比較用に生成)
    psfs_no_mask = gen_psfs_no_mask(oof_phase, aperture)
    psfs_with_mask = gen_psfs_with_mask(oof_phase, cpm_phase, aperture)
    print(f"PSFスタック(マスクなし): shape={psfs_no_mask.shape}")
    print(f"PSFスタック(CPMあり)  : shape={psfs_with_mask.shape}")

    # 保存
    np.save(OUTPUT_DIR / "psfs_with_mask.npy", psfs_with_mask)
    np.save(OUTPUT_DIR / "phi_values.npy", phi_values)
    print(f"\nNumPy配列を保存 : {OUTPUT_DIR / 'psfs_with_mask.npy'}")

    # 可視化
    plot_psf_comparison_grid(psfs_no_mask, psfs_with_mask, phi_values,
                              OUTPUT_DIR / "psf_comparison_grid.png")
    print(f"図を保存        : {OUTPUT_DIR}/psf_comparison_grid.png")

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

    return psfs_no_mask, psfs_with_mask, phi_values, ncc_no_mask, ncc_with_mask


if __name__ == "__main__":
    psfs_no_mask, psfs_with_mask, phi_values, ncc_no_mask, ncc_with_mask = main()
