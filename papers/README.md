# papers/ — LaTeXビルドメモ

卒論・進捗報告など、執筆中のLaTeX文書を置くディレクトリ。

## ビルドコマンド

ファイル名は明示すること。引数なしの `latexmk` はカレントディレクトリの
`\begin{document}` を含む `.tex` を全部拾ってビルドするため、`.tex` が複数
になった時点で意図しないファイルまで一緒にビルドされうる。

```bash
cd papers && latexmk progress_report.tex      # 通常ビルド(xelatex → biber → xelatex を自動で必要回数)
cd papers && latexmk -c progress_report.tex   # 中間生成物(aux/log/bbl等)だけ削除、PDFは残す
cd papers && latexmk -C progress_report.tex   # PDFも含めて完全に削除
cd papers && latexmk -pvc progress_report.tex # 保存を検知して自動リビルド + Previewで表示(Ctrl+Cで終了)
```

設定は [.latexmkrc](.latexmkrc)。エンジンはXeLaTeX固定(日本語フォント・biblatexを使うため)。

## 環境

- TeX Live 2025(`/usr/local/texlive/2025`)。pdflatex/xelatex/lualatex/latexmk/biber インストール済み。
- 日本語フォントはローカル環境(Hiragino、macOS標準)とOverleaf等(Noto CJK)を `\IfFontExistsTF` で自動判定しているので、`.tex` 側は環境によらず手直し不要([progress_report.tex](progress_report.tex) 冒頭参照)。

## 使える主なパッケージ

新しい `.tex` を作る際は `progress_report.tex` の プリアンブルをベースにコピーすればよい。含まれるもの:

| パッケージ | 用途 |
|---|---|
| `siunitx` | 物理量の単位表記(`\SI{1.55}{\micro\meter}` など) |
| `physics` | 光学・偏微分の数式表記 |
| `subcaption` | 複数パネル図(PSF比較グリッドなど) |
| `tikz` | 光学系の概念図・模式図 |
| `cleveref` | `\cref` で図表番号を自動判別 |
| `biblatex` + `biber` | 文献管理(`references.bib` を参照) |

## 文献管理

[references.bib](references.bib) にBibTeXエントリを追加し、本文で `\cite{key}`、末尾に `\printbibliography`。
`report_chang` エントリは著者情報未確認の仮登録なので、判明次第更新すること。

## ディレクトリ構成

- `figures/`: 図版(png等)
- `references.bib`: 文献データベース
- `build/`: `latexmk`実行時に生成される中間ファイル(.aux/.log/.bbl/.bcf等)置き場。`.latexmkrc`で`$aux_dir='build'`に設定済みなので、`papers/`直下は`.tex`とPDF(と`.synctex.gz`)だけで綺麗に保たれる。`.gitignore`で無視されるので、削除しても`latexmk`一発で再生成される。
