#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

config=$1
shift

block_pointers_via_raising=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --raise-block-pointers)
      shift
      block_pointers_via_raising=1
      ;;
    --external-pad)
      shift
      export PREPROCESS_EXTERNAL=1
      ;;
    --datatype-bf16)
      shift
      export DATATYPE=bf16
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      exit 1
      ;;
  esac
done


if [ "$config" = "xsmm-scalar" ]; then
  export TRITON_CPU_TRITON_XSMM=1
elif [ "$config" = "xsmm-block" ]; then
  if [ $block_pointers_via_raising = 1 ]; then
    export TRITON_CPU_RAISE_BLOCK_POINTER=1
  else
    export USE_BLOCK_POINTERS=1
  fi
  export TRITON_CPU_TRITON_XSMM=1
elif [ "$config" = "xsmm-pad-k" ]; then
  if [ $block_pointers_via_raising = 1 ]; then
    export TRITON_CPU_RAISE_BLOCK_POINTER=1
  else
    export USE_BLOCK_POINTERS=1
  fi
  echo "NOTE: requires that \$XSMM_LIB_DIR is in LD_LIBRARY_PATH"
  export XSMM_PAD=1
  export TRITON_CPU_TRITON_XSMM=1
elif [ "$config" = "xsmm-loop-collapse-pad-b" ]; then
  export PAD_B_ONLY=1
  export TRITON_CPU_LOOP_BRGEMM_XSMM=1
  if [ $block_pointers_via_raising = 1 ]; then
    export TRITON_CPU_RAISE_BLOCK_POINTER=1
  else
    export USE_BLOCK_POINTERS=1
  fi
  echo "NOTE: requires that \$XSMM_LIB_DIR is in LD_LIBRARY_PATH"
  export XSMM_PAD=1
elif [ "$config" = "xsmm-external-pad" ]; then
  echo "NOT A TRUE CONFIG (... right?); try --external-pad on another config"
  exit 1
else
  echo "ERROR: unrecognized config: $config"
  exit 1
fi

python $SCRIPT_DIR/03-matrix-multiplication-cpu.py


