# Load balancing: TODO

Status: planning document for the `load_balancing` branch.
Scope: streaming / in-situ workflows only; file-based checkpoint workflows are out of scope.

## 1. Deduplicate the output granularity strategies

### Current state

The particle output side contains three granularity strategies, implemented as
near-copies with per-copy changes:

| Granularity | Chunk construction | Particle writer | Patch metadata writer |
|---|---|---|---|
| One patch per rank | implicit single range | `writeParticlesWithoutAnySubdividing` | `writeParticlePatches` |
| One patch per supercell | `createAChunkForEverySupercell` | `writeParticlesBySupercells` | `writeParticlePatchesBySupercells` |
| Size-bounded patches (compromise) | `createSupercellRangeChunks` | `writeParticlesByChunks` | `writeParticlePatches` |

All three share the same surrounding work: particle counting, global offsets,
all-gathers, strategy setup, IO rounds, flush/logging, empty-iteration handling,
and patch metadata. That logic is currently duplicated across
`include/picongpu/plugins/openPMD/WriteSpecies.hpp` and
`include/picongpu/plugins/openPMD/writer/misc.hpp`.

### Tasks

- [ ] Introduce an explicit strategy parameter, e.g.
      `enum class PatchGranularity { PerRank, PerSupercell, SizeBounded };`.
- [ ] Merge chunk construction into one implementation parameterized by
      granularity:
      - `PerRank`: one range covering the whole local domain.
      - `PerSupercell`: one range per supercell.
      - `SizeBounded`: greedy packing of consecutive supercells up to
        `--openPMD.particleIOChunkSize` (with the existing warning/limit
        increase for a single over-sized supercell).
- [ ] Merge the particle-writing loop into one implementation parameterized by
      the strategy.
- [ ] Merge patch metadata writing into one implementation. Keep the one
      per-strategy constraint (`PerSupercell` requires the same number of
      patches on every MPI rank) as a validation step, not as a separate copy.
- [ ] Keep all observable output identical for each strategy:
      patch count, `beginSupercellIdx`/`endSupercellIdx`, `numParticles`,
      `numParticlesOffset`, `offset`, `extent`, and iteration layout.
- [ ] Make the strategy selectable from configuration/CLI.
- [ ] Add tests comparing all three strategies, including zero-particle
      supercells, ranks with different local supercell counts, and one
      over-sized supercell.
- [ ] Verify that restart (`LoadSpecies`) consumes output from all three
      strategies.

## 2. Proper wiring of the on-the-fly restart operation

### Current state

`Checkpoint::checkpoint()` in
`include/picongpu/plugins/Checkpoint.x.cpp` currently abuses the regular
checkpoint entry point:

- if `MPI_Query_thread()` reports `MPI_THREAD_MULTIPLE`, it starts
  `doRestart()` asynchronously while the checkpoint is being written;
- otherwise it only writes the checkpoint.

This means a normal `--checkpoint.period` can silently restart the simulation.
That was useful as a test harness but must not become the production interface.

### Tasks

- [ ] Separate the concepts:
      - checkpoint = persist state;
      - on-the-fly restart/reload = consume freshly written state for
        rebalancing.
- [ ] Introduce a dedicated operation/plugin or backend API for
      checkpoint-and-reload (e.g. `checkpointAndReload()` or a
      `LoadBalance` controller) instead of overloading `Checkpoint::checkpoint`.
- [ ] Move ownership of the writer/reader synchronization state machine from
      the generic checkpointing call into the dedicated operation.
- [ ] Define backend capabilities explicitly:
      - supports concurrent read-while-write (streaming/SST),
      - does not support it (must fail clearly or use a non-concurrent path).
- [ ] Make the trigger explicit and configurable; regular checkpointing must
      never restart.
- [ ] Keep failure semantics explicit: reader failure releases the writer and
      is propagated (already partially implemented), writer failure must not
      leave the reader hanging.
- [ ] Add tests:
      - checkpoint-only does not reload;
      - explicit on-the-fly operation reloads exactly once;
      - failed reload aborts with an error instead of deadlocking.

## 3. Actual load balancing

The current branch only restarts into the same decomposition: the freshly
written checkpoint is read back into the same local domains. That is a no-op
and is only a prerequisite for real rebalancing.

### Constraints to respect

- PIConGPU's field solver, guard exchange, and current deposition assume one
  contiguous Cartesian subdomain per rank. An arbitrary patch-to-rank mapping
  is not supported without a major field/decomposition refactor.
- Initial rebalancing should therefore compute a new **rectangular**
  decomposition (contiguous blocks), and use the existing patch-aware restart
  to load old data into the new blocks.
- Patch metadata (per supercell or size-bounded) provides the particle counts
  needed to estimate where work is.
- Streaming/in-situ transport is the target; filesystem checkpoints are out of
  scope.

### Possible rebalancing strategies (explored)

1. **Static `--gridDist` (baseline)**
   - Predefined, non-uniform distribution of cells to ranks.
   - Pros: already exists, zero runtime cost.
   - Cons: not adaptive; no response to evolving particle load.

2. **Weighted axis cuts / weighted median**
   - Compute cumulative per-supercell weights along one axis and place cut
     points so every rank receives approximately equal weight.
   - Pros: simple, keeps contiguous rectangular blocks, no external
     dependencies, easy to seed from patch counts.
   - Cons: one-dimensional only; poor for multi-dimensional imbalance.

3. **Recursive coordinate bisection (RCB) / geometric partitioning**
   - Recursively split the bounding box along the axis with the largest
     weighted extent (or use Zoltan RCB).
   - Pros: handles multi-dimensional imbalance, produces contiguous-ish
     blocks, well understood.
   - Cons: may need Zoltan or an in-house implementation; cuts can be less
     communication-optimal than graph-based methods.

4. **Space-filling curve (Morton/Hilbert) partitioning**
   - Order supercells by an SFC, cut the curve into equal-weight ranges.
   - Pros: locality-preserving, naturally weight-aware, works in arbitrary
     dimensions.
   - Cons: resulting blocks are generally not rectangular; mapping them back
     to rectangular subdomains or adapting the field decomposition is
     non-trivial.

5. **Graph partitioning (ParMETIS / PT-Scotch / Zoltan)**
   - Build a patch/supercell adjacency graph with particle weights and edge
     communication weights; partition into balanced contiguous parts.
   - Pros: can optimize both balance and communication volume.
   - Cons: external dependency, partition quality/contiguity constraints,
     setup cost, harder to integrate with the rectangular decomposition
     assumption.

6. **Diffusive / neighbor-based boundary shifting**
   - Iteratively move domain boundaries between neighboring ranks based on
     measured weight differences.
   - Pros: keeps the rectangular decomposition, low communication, easy to
     implement incrementally, naturally accounts for local imbalance.
   - Cons: slow convergence, may get stuck in local optima, requires repeated
     small re-decompositions or a final global layout.

7. **Work stealing / dynamic patch queues**
   - Treat patches as tasks pulled by idle ranks.
   - Pros: very adaptive.
   - Cons: fundamentally incompatible with the current fixed field
     decomposition and guard/current-deposition model; would require a major
     redesign.

8. **Particle-only migration with fixed fields**
   - Keep the field decomposition and migrate particles between ranks.
   - Pros: no field reallocation.
   - Cons: particles would leave the local field domain needed for gather and
     deposition; not viable without redesigning the particle-to-domain
     coupling.

### Recommended direction

- Start with a contiguous, dependency-free method based on patch weights:
  weighted axis cuts / recursive bisection, optionally refined by neighbor
  boundary shifting.
- Use the existing in-situ checkpoint + patch-aware restart as the migration
  mechanism: compute a new rectangular decomposition, then reload old patches
  into the new blocks.
- Revisit SFC or graph partitioning only after the trigger, weight model, and
  restart wiring are stable.

### Tasks

- [ ] Define the load metric:
      - per-patch/supercell particle count as a first proxy;
      - add cost weights for particle push/current deposition and field/guard
        exchange;
      - optionally calibrate with measured per-step timings.
- [ ] Define the trigger:
      - periodic evaluation with hysteresis;
      - imbalance threshold;
      - avoid rebalancing on every step/checkpoint.
- [ ] Implement weight collection and global reduction.
- [ ] Implement the chosen decomposition algorithm (start with weighted
      recursive bisection or weighted axis cuts).
- [ ] Replace the current self-restart target decomposition with the computed
      decomposition.
- [ ] Decide and document how fields are handled on re-decomposition
      (reload vs redistribute).
- [ ] Add validation:
      - synthetic imbalanced particle distributions;
      - no-op equivalence (same decomposition produces identical physics);
      - load imbalance improves after rebalancing;
      - restart correctness and absence of deadlocks.
- [ ] Document limitations: only rectangular contiguous decompositions are
      supported initially.
