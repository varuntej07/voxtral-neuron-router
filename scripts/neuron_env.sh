# Environment for the trn1 training run. Source it, do not execute it.
#
#     bash scripts/setup_train_venv.sh     # once per instance
#     source scripts/neuron_env.sh
#
# These have to be set before torch_xla is imported, because the compiler flags are
# read at graph compile time and the cache path is read when the runtime starts.

# The DLAMI venvs cannot run this stack (see ami-has-no-tracing-path in lessons.txt), so
# training uses its own venv. Activating it is not optional: torch_neuronx shells out to
# libneuronpjrt-path from the venv bin directory, which calling venv/bin/python skips.
source "$HOME/venv-train/bin/activate"

# --target=trn1 is named rather than inferred, the same reason as on inf2: the artifact
# should not depend on which box ran the compiler. --model-type=transformer turns on the
# transformer specific passes. --enable-saturate-infinity clamps inf to the dtype max,
# which keeps a bf16 overflow in the backward pass from turning into NaN.
export NEURON_CC_FLAGS="--target=trn1 --model-type=transformer --enable-saturate-infinity"

# Compiled graphs are keyed by shape, so this cache is what keeps a rerun from
# recompiling. Keep it on the instance disk and keep it across runs.
export NEURON_COMPILE_CACHE_URL="$HOME/neuron_cache"

# NEURON_RT_NUM_CORES is deliberately not set. Under torchrun each worker should take one
# NeuronCore of the two on trn1.2xlarge; a per-process count of 2 can make the second
# worker fail to get a core. Confirm the assignment with neuron-top during the smoke run.

# CPU fallback is not surfaced by XLA_IR_DEBUG or XLA_HLO_DEBUG; those only attach Python
# source locations to the IR, which makes compiler logs readable. The fallback shows up
# as aten:: counters in torch_xla.debug.metrics, which train_lora.py writes to --results.
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

# Long compiles on a fresh cache are normal; this stops the runtime giving up on them.
export NEURON_RT_EXEC_TIMEOUT=600

# The root volume filled on inf2 partly from the pip cache (root-volume-fills-fast).
export PIP_NO_CACHE_DIR=1

# Recorded into the results JSON, so a step time is never read without its box.
export NEURON_INSTANCE_TYPE="${NEURON_INSTANCE_TYPE:-trn1.2xlarge}"
