# SNN_Transform R1 — Embodied Sparse Event Propagation

## Starting point

The frozen source package is `SNN_Dynamical_Transformer_Core_FROZEN_20260922`. R0 is **not rerun** here; its stored audit result is retained as the reference checkpoint.

R0 already closed temporal K-Q relation, recurrent selection, autonomous STOP, multi-hop routing and independent hard Value retrieval, but graph traversal after a route event was executed by a Python `current -> children.gather -> next_node` frontier scheduler.

## R1 scientific change

R1 changes only route propagation:

```text
local child route spike
    -> fixed sparse structural synapse
    -> receiver-node activation spike
    -> receiver local circuit
```

The routing packet is represented by distributed `node_state[B,N]`. Child-route populations are presynaptic sources indexed by `(sender_node, child_slot)`. A fixed sparse COO projection maps those route spikes to the postsynaptic receiver node.

The forward route contains no `current` node variable, selected-child `gather`, `argmax`, `topk`, or `softmax`. Temporal relation and recurrent competition are intentionally unchanged from the frozen core.

## R1 gate

R1 passes only if:

- source audit confirms the old frontier traversal is absent;
- sparse synaptic projection and distributed node state are present;
- depth 1–6 node/STOP/full-path/Value accuracy remain 100%;
- a target at depth `d` emits exactly `d` child-route spikes and one STOP spike;
- projection synaptic-event count equals emitted child-route spike count;
- five unseen structural permutations remain 100%;
- the frozen noise range through sigma=0.01 remains 100%.

## Boundary

Passing R1 closes the **software algorithmic traversal** gap. It does not yet prove SpiNNaker timing/energy equivalence, graded multi-Value aggregation, learned topology/representation, or generic non-hierarchical attention.
