#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source ../miniforge/bin/activate triton

MULTITHREAD_THREADS=56  # SPR: 56

export LD_PRELOAD=/lib64/libomp.so:$LD_PRELOAD
# Hyper-Threading
export KMP_AFFINITY=granularity=fine,compact,1,0
# No Hyper-Threading
#export KMP_AFFINITY=granularity=core,compact,0,0

#for datatype in f32; do
for datatype in bf8 bf16 f32; do
  for num_threads in $MULTITHREAD_THREADS 1; do
    export DATATYPE=$datatype
    export TRITON_CPU_MAX_THREADS=${num_threads}
    export OMP_NUM_THREADS=${num_threads}

    export USE_BLOCK_POINTERS=1
    echo -e "\n\nRUN: triton-cpu-baseline-block | threads $num_threads | type $datatype"
    time python $SCRIPT_DIR/python/tutorials/03-matrix-multiplication-cpu.py 2>&1

    export USE_BLOCK_POINTERS=0
    echo -e "\n\nRUN: triton-cpu-baseline-scalar | threads $num_threads | type $datatype"
    time python $SCRIPT_DIR/python/tutorials/03-matrix-multiplication-cpu.py 2>&1
  done
done

conda deactivate
