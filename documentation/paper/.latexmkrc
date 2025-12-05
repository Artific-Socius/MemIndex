# 设置输出目录为 build
$out_dir = 'build';

# 确保 build 目录存在（跨平台兼容）
unless (-d 'build') {
    mkdir 'build' or die "Cannot create build directory: $!";
}

# PDF 输出设置 - 使用 PDFLaTeX（配合 CJK 包支持中文）
$pdf_mode = 4;  # 4 = pdflatex (使用 pdfLaTeX 配合 CJK 包)
$postscript_mode = 0;
$dvi_mode = 0;

# 同步设置 - PDFLaTeX
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=build %O %S';
# 保留 XeLaTeX 设置以备需要时使用
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=build %O %S';
$pdf_previewer = 'start';

# 清理设置
$clean_ext = 'bbl synctex.gz synctex(busy) fdb_latexmk fls';

# 自动清理辅助文件
$clean_full_ext = 'aux bbl blg idx ind lof lot out toc acn acr alg glg glo gls fls log fdb_latexmk snm nav synctex.gz synctex(busy) pdfsync';

