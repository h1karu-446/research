#!/usr/bin/env perl
# XeLaTeX + biber ビルド設定 (papers/ 以下で `latexmk` を実行するだけでよい)
$pdf_mode = 5;          # 5 = xelatex
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$bibtex_use = 2;
$biber = 'biber %O %S';
$pdf_previewer = 'open -a Preview %O %S';

# 中間生成物(.aux/.log/.bbl/...)は build/ に隔離し、papers/ 直下を汚さない。
# 最終成果物の PDF/synctex だけは papers/ 直下(カレント)に残す。
$aux_dir = 'build';
$out_dir = '.';
$ENV{'TEXMFOUTPUT'} = 'build';

# 中間生成物も含め、対象ファイル名の一括削除対象に追加(`latexmk -c` 用)
push @generated_exts, 'synctex.gz', 'bbl', 'bcf', 'run.xml';
