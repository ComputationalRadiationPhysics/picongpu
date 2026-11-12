# Load balancing: TODO

Status: planning document for the `load_balancing` branch.
Scope: streaming / in-situ workflows only; file-based checkpoint workflows are out of scope.

## 0. Context and status (from PR #5405)

PR #5405, "WIP: Prototype for 'poor man's' load balancing", describes the
overall idea:

1. Halt the simulation.
2. *Virtually* write a checkpoint.
3. Restart from that same checkpoint while repartitioning the domain, i.e. load
   the data into a more balanced decomposition.

The "virtual checkpoint" uses the openPMD streaming backend, so no disk I/O is
needed: the data is streamed to the same application and read back with a
different domain decomposition. In-situ restarting and load-balanced
restarting are orthogonal features that both feed into this use case.

Status noted in the PR:

- The "restart with changed domain decomposition" part has largely been merged
  (PR #5508); the rest remains a proof of concept.
- The branch is intended to be maintained only if on-the-fly rebalancing is
  pursued.
- At prototype stage, partial patches were ignored; granularity below the
  supercell is considered unnecessary for load balancing for now.
- The prototype relies on full `MPI_THREAD_MULTIPLE` support and on the SST
  engine.
- There is no proper CLI control for the virtual checkpoint yet; the regular
  `--checkpoint.period` is abused for testing.

The sections below extend the original three-point plan with the concrete
issues and scaling ideas from that PR.

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
- [ ] Replace the ad-hoc use of `--checkpoint.period` with a dedicated trigger
      and CLI/config options for the virtual checkpoint/rebalance operation.
- [ ] Investigate whether full `MPI_THREAD_MULTIPLE` is really required, or
      whether SST startup and the writer/reader handshake can be controlled
      more finely (e.g. with serialized MPI or a dedicated communicator).
- [ ] Do not reload `idProvider` and RNG state during an on-the-fly reload:
      a rebalance must keep the running simulation's IDs and RNG stream, unlike
      a real restart from a checkpoint.
- [ ] Define handling for PML fields (ignored in the prototype) and inventory
      other plugins that carry checkpoint/restart state (not considered yet).
- [ ] Document the working SST configurations (see also section 5), including
      `DataTransport`, fabric/UCX environment variables, and startup ordering.
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

- [ ] Decide where the new decomposition is computed and how it is applied.
      The PR explicitly lists this as open: there is no canonical answer for the
      new decomposition, and changing the simulation's decomposition from
      within the IO plugin needs a proper interface (IO plugin vs.
      simulation/PMacc control).
- [ ] Define a decomposition handoff format (per-rank block offsets/extents or
      cut positions) shared between the load balancer, the IO/reload layer, and
      the simulation control.
- [ ] Confirm the minimum useful granularity. The PR assumes supercell-level
      patches are sufficient for load balancing; below-supercell splitting is
      explicitly out of scope for now.
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

## 4. Scaling and follow-ups from PR #5405

### Particle patch communication and metadata

- [ ] Address the N->N communication pattern of particle patches: every writer
      writes patches and every reader reads all of them. This is acceptable for
      file-based checkpoints (files aggregate), but in streaming it is not.
      Options listed in the PR:
      - write particle patches to a parallel filesystem instead of streaming
        them (out of scope here, but keep as a fallback idea);
      - aggregate/deaggregate patches;
      - use ADIOS2 Local Values: metadata aggregation with one value per
        process, which fits a two-level patch hierarchy.
- [ ] Design a two-level patch hierarchy to control the number of patches:
      - lower level: one patch per supercell (fine-grained for rebalancing);
      - higher level: one patch per process, pointing into the lower-level
        patches rather than into particle data;
      - only the small higher level needs N->N communication and can be stored
        as an ADIOS2 Local Value.
- [ ] Merge small I/O operations while keeping the logical patch structure:
      the prototype performs one I/O operation per supercell, which will not
      scale to production networks. Read and write should batch ranges while
      preserving particle patches as the logical granularity.
- [ ] Revisit the size-bounded granularity as the likely default once the
      two-level hierarchy exists: it already reduces the number of I/O
      operations while patch metadata can still be refined separately.

### Memory consumption

- [ ] Account for SST host-side streaming buffers: they hold roughly a CPU
      clone of the GPU simulation volume plus particles. Define a memory budget
      and validate it for production-sized runs.
- [ ] Investigate whether the ADIOS2 BP5 serializer used by SST can use node
      SSDs as a streaming buffer instead of RAM (this is an ADIOS2-side
      project; memmap/RDMA interaction is unclear).
- [ ] Until then, document workarounds:
      - leave headroom in GPU memory instead of filling it completely;
      - use a disk checkpoint if full GPU memory utilization is required
        (file-based workflow, out of scope here).

### Other state and fields

- [ ] Define how PML fields, which have non-rectangular geometry, participate
      in rebalancing; the prototype ignores them.
- [ ] Inventory other plugins that keep checkpoint/restart state and decide
      which must be reloaded during an on-the-fly rebalance.
- [ ] Clarify whether fields are reloaded or redistributed on re-decomposition,
      and whether field data is included in the virtual checkpoint or kept in
      place.

## 5. Working SST configurations (from PR #5405)

The PR notes that SST requires system-specific transport configuration. The
option names below use the current `backendConfig` spelling; the PR predates
that rename and used `--checkpoint.openPMD.json`.

Fabric transport:

```bash
FABRIC_PROVIDER=shm FABRIC_IFACE=shm \
PIC_USE_THREADED_MPI=MPI_THREAD_MULTIPLE \
picongpu -s 100 -g 32 32 32 -d 1 1 1 \
  --checkpoint.period 50:50 \
  --checkpoint.openPMD.ext sst \
  --checkpoint.openPMD.backendConfig \
    '{"adios2":{"engine":{"parameters":{"DataTransport":"fabric"}}}}'
```

UCX transport:

```bash
UCX_TLS=shm UCX_POSIX_USE_PROC_LINK=n \
PIC_USE_THREADED_MPI=MPI_THREAD_MULTIPLE \
picongpu -s 100 -g 32 32 32 -d 1 1 1 \
  --checkpoint.period 50:50 \
  --checkpoint.openPMD.ext sst \
  --checkpoint.openPMD.backendConfig \
    '{"adios2":{"engine":{"parameters":{"DataTransport":"ucx"}}}}'
```

In this repository, `DataTransport=MPI` was also verified to work for the
on-the-fly path, and is the least environment-dependent option on a single
node:

```json
{"adios2":{"engine":{"parameters":{"DataTransport":"MPI","OpenTimeoutSecs":"10"}}}}
```

- [ ] Formalize the transport/startup requirements in documentation.
- [ ] Investigate finer control of SST startup so that fully threaded MPI is
      not a hard requirement.
