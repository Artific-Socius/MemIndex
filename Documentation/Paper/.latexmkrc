# 设置输出目录为 build
$out_dir = 'build';

# 确保build目录存在
system("mkdir -p build") unless -d 'build';

# 设置PDF查看器（macOS使用open命令）
$pdf_previewer = 'open';

# 确保LaTeX能找到源文件中的图片和类文件
# 通过设置TEXINPUTS环境变量，让LaTeX在源目录中查找文件
$ENV{'TEXINPUTS'} = './/:' . ($ENV{'TEXINPUTS'} // '');

# 使用pdflatex编译
$pdf_mode = 1;

# 清理时也清理build目录中的文件
$clean_ext = 'bbl synctex.gz';

# 设置bibtex和biber的输出目录
$bibtex_use = 2;
$biber = 'biber --output-directory=build %O %S';
