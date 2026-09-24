# Environment for the trn1 training run. Source it, do not execute it.
#
#     source scripts/neuron_env.sh
#
# These have to be set before torch_xla is imported, because the compiler flags are
# read at graph compile time and the cache path is read when the runtime starts.

# --model-type=transformer turns on the transformer specific passes in neuronx-cc.
# Without it the compiler treats the attention blocks as generic matmuls and both
# compile time and step time get noticeably worse.
export NEURON_CC_FLAGS="--model-type=transformer --enable-saturate-infinity"

# Compiled graphs are keyed by shape, so this cache is what keeps a rerun from
# recompiling. Keep it on the instance disk and keep it across runs.
export NEURON_COMPILE_CACHE_URL="$HOME/neuron_cache"

# trn1.2xlarge exposes one Trainium device with two NeuronCores.
export NEURON_RT_NUM_CORES=2

# XLA falls back to CPU silently when it meets an op it cannot lower, which turns a
# fast run into a slow one with no error. This makes that fallback visible.
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

# Long compiles on a fresh cache are normal; this stops the runtime giving up on them.
export NEURON_RT_EXEC_TIMEOUT=600
