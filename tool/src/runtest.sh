#!/usr/bin/env bash
#export MKL_THREADING_LAYER=gnu
export GEOMSTATS_BACKEND=tensorflow
projectdir=`echo $1 | grep -o ".*/projects/[a-zA-Z0-9_]\+"`
testfile=$1
testclass=$2
testname=$3
envname=$4
echo $projectdir
echo $testfile
echo $testclass
echo $testname
echo $envname

source /home/cc/miniconda/etc/profile.d/conda.sh
#source ~/miniconda3/etc/profile.d/conda.sh

which conda
# make the virtual environment name parameterizable
conda activate ${envname}
conda env list

cd $projectdir
echo $projectdir
if [[ ${testclass} == "none" ]]; then
    #pytest -W ignore --capture=no ${testfile}::${testname}
    taskset -c $5 python -m pytest -W ignore --capture=no ${testfile}::${testname}
    #python -m pytest -W ignore --capture=no ${testfile}::${testname}
    retcode=$?
else
    #pytest -W ignore --capture=no ${testfile}::${testclass}::${testname}
    taskset -c $5 python -m pytest -W ignore --capture=no ${testfile}::${testclass}::${testname}
    retcode=$?
fi
conda deactivate
cd -
exit $retcode

