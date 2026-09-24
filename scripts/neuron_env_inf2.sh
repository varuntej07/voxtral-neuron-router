# Environment for the inf2 serving box. Source it, do not execute it.
#
#     source scripts/neuron_env_inf2.sh
#
# This is the inference twin of neuron_env.sh. It is a separate file because the two
# jobs want different things: training wants saturate-infinity and two cores of
# collectives, inference wants exact precision and one core per graph.

# --model-type=transformer turns on the transformer specific passes. --auto-cast=none
# is the important one here: by default neuronx-cc downcasts fp32 matmuls to bf16 on
# its own, which would mean a parity number that measures the compiler's dtype choice
# instead of the hardware. The weights are already bf16, so nothing is lost by saying no.
#
# --target=inf2 is named rather than inferred. This is an Inferentia2 box, NeuronCore-v2,
# and the compiler picks a default from the instance it happens to be running on. Saying
# it out loud means a graph built here is the graph that serves here, and it means the
# same command produces the same artifact if it is ever run from a trn box.
export NEURON_CC_FLAGS="--target=inf2 --model-type=transformer --auto-cast=none"

# One graph per shape, so this cache is the difference between a 30 minute rerun and a
# 30 second one. Keep it on instance disk and keep it across runs.
export NEURON_COMPILE_CACHE_URL="$HOME/neuron_cache"

# inf2.xlarge is one Inferentia2 chip, which is two NeuronCores. A traced model takes
# one core unless it was compiled for more, so both stages can be loaded at once later.
export NEURON_RT_NUM_CORES=2

# Compiling a 3B graph on four vCPUs is slow. This stops the runtime giving up while it
# waits, and stops the first cold load being reported as a hang.
export NEURON_RT_EXEC_TIMEOUT=600

# Recorded into results/neuron_parity.json so a latency number can never be read back
# without knowing which box produced it.
export NEURON_INSTANCE_TYPE="${NEURON_INSTANCE_TYPE:-inf2.xlarge}"
