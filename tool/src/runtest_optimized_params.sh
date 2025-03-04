#!/usr/bin/env bash
#export MKL_THREADING_LAYER=gnu
#export GEOMSTATS_BACKEND=tensorflow

projectdir=`echo $1 | cut -d"/" -f1-8`
testfile=$1
testclass=$2
testname=$3
envname=$4
echo $testfile
echo $testclass
echo $testname
echo $envname

source /home/cc/miniconda/etc/profile.d/conda.sh
#source ~/miniconda3/etc/profile.d/conda.sh

# make the virtual environment name parameterizable
conda activate ${envname}

cd $projectdir

if [[ ${testclass} == "none" ]]; then
    #seq 1 $threadcount | parallel --lb -n0 time pytest -W ignore --capture=no ${testfile}::${testname}

    python3 -m pytest -W ignore --capture=no -v ${testfile}::${testname} | tee >(grep FAILED | wc -l) >(grep PASSED | wc -l) | paste -sd "," -
    retcode=$?
else
    #seq 1 $threadcount | parallel --lb -n0 time pytest -W ignore --capture=no ${testfile}::${testclass}::${testname}

    python3 -m pytest -W ignore --capture=no -v ${testfile}::${testclass}::${testname} | tee >(grep FAILED | wc -l) >(grep PASSED | wc -l) | paste -sd "," -
    retcode=$?
fi
conda deactivate
cd -
exit $retcode