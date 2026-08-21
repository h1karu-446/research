"""
v0.3: Wienerデコンボリューションによる EDOF (Extended Depth of Field) 実証
==================================================
v0.2b(experiments/v0_2/v0_2_cubic_phase_mask_aliasing_fixed.py)が生成した
エイリアシング修正済みのPSFスタック(no mask / CPM, 21デフォーカス断面)を
読み込み、Wienerデコンボリューションによる EDOF を実証する。
PSF自体の生成・エイリアシング対策はv0.2bの担当であり、本ファイルは
それを既知の入力として扱う(責務を分離するため、v0.3ではPSFを再計算しない)。

report_chang.pdf の主張:「合焦時に測定/計算した単一のCPM PSFでデコンボリューム
すれば、デフォーカスしていても特徴を復元できる」を、合成テストターゲット
(チャープパターン)で数値的に検証する。比較対象として、マスクなし
(clear aperture)を同じプロトコル(合焦PSF固定・Wiener復元)で処理した
場合の性能も示す。

結果(outputs/psnr_vs_defocus.png)は単調ではない:
CPMはkWm=±1近傍でno-maskを大きく上回る(PSNR +5.5dB)が、
1<|kWm|<=5では逆にno-maskをやや下回り、|kWm|=10近傍で再び同程度に
収束する。これはno-maskの「間違ったカーネルでも無難に平坦な出力になる」
性質と、CPMの「間違ったカーネルだと構造化された誤り(自信満々な誤答)に
なりうる」性質の差によるものと考えられ、report_chang.pdf自身も
Wienerフィルタのリンギングと、CPM不変性が有限範囲(実測で±30〜50μm)
であることに言及している(main()末尾の出力と詳細コメントを参照)。
一方 v0.2b で計算した形状不変性の指標(NCC)では、|kWm|>=6でも
CPMがno-maskを明確に上回っており(0.48 vs 0.04)、PSNRの逆転は
「デコンボリューション特有」の効果であることが示唆される。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# 1. パラメータ定義・入力データの読み込み
# =============================================================================

V02_OUTPUT_DIR = Path(__file__).parent.parent / "v0_2" / "outputs"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_psf_stacks():
    """v0.2b(エイリアシング修正版)が生成したPSFスタックを読み込む。"""
    psfs_no_mask = np.load(V02_OUTPUT_DIR / "psfs_no_mask_aliasing_fixed.npy")
    psfs_with_mask = np.load(V02_OUTPUT_DIR / "psfs_with_mask_aliasing_fixed.npy")
    phi_values = np.load(V02_OUTPUT_DIR / "phi_values_aliasing_fixed.npy")
    return psfs_no_mask, psfs_with_mask, phi_values


# =============================================================================
# 2. Wienerデコンボリューション
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
# 3. 合成テストターゲット(チャープ解像度パターン)
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
# 4. 可視化
# =============================================================================
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
# 5. メイン
# =============================================================================
def main():
    print("=" * 60)
    print("v0.3: Wienerデコンボリューション(EDOF実証)")
    print("=" * 60)

    psfs_no_mask, psfs_with_mask, phi_values = load_psf_stacks()
    n_phi = phi_values.shape[0]
    crop_half = psfs_no_mask.shape[1] // 2
    print(f"PSFスタックを読み込み(v0.2b由来): shape={psfs_no_mask.shape}")

    # ターゲット画像はカーネル(2*crop_half)より十分大きく取る
    # (embed_kernelがカーネルを画像内に収める必要があるため)
    target = gen_chirp_target(size=2 * crop_half + 200)
    focus_idx = n_phi // 2  # kWm=0

    kernel_nm_focus = normalize_kernel(psfs_no_mask[focus_idx])
    kernel_cpm_focus = normalize_kernel(psfs_with_mask[focus_idx])

    rng = np.random.default_rng(0)
    NOISE_SIGMA = 0.01
    K_WIENER = 3e-3

    blur_nm = np.empty((n_phi, *target.shape), dtype=np.float32)
    blur_cpm = np.empty_like(blur_nm)
    rec_nm = np.empty_like(blur_nm)
    rec_cpm = np.empty_like(blur_nm)
    psnr_nm = np.empty(n_phi)
    psnr_cpm = np.empty(n_phi)

    for j in range(n_phi):
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
    #     (v0.2bのNCC指標が示す通り)。しかし合焦カーネルとの不一致が大きい
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
    print(f"  v0.2bのNCC指標(形状不変性)ではCPMが|kWm|>=6でも優位(0.48 vs 0.04)。")
    print(f"  PSNRの逆転はデコンボリューション特有の効果であり、次段で要検討。")

    return phi_values, psnr_nm, psnr_cpm


if __name__ == "__main__":
    main()
