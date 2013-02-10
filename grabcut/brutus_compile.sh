mcc -R -singleCompThread -R -nodisplay -R -nojvm -m grabcutMain.m -a mosek -a matluster -a svmstruct -a svmstruct/helpers -a mpe_inference/bk -a mpe_inference/ibfs -a helpers
sed -i '7i\export MCR_CACHE_ROOT=${TMPDIR};' run_grabcutMain.sh
sed -i '26i\LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$HOME/mosek/6/tools/platform/linux64x86/bin' run_grabcutMain.sh
