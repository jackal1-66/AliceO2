# AOD Merger — Complete Technical Reference

This document describes the design, implementation, and execution model of
`aodMerger.cxx`, the standalone tool that merges ALICE O2 Analysis Object Data (AOD)
files. It is written as a self-contained reference for readers who are not already
familiar with the codebase.

---

## Table of Contents

1. [Background — what are AO2D files?](#1-background--what-are-ao2d-files)
2. [What the merger must do](#2-what-the-merger-must-do)
3. [Command-line interface](#3-command-line-interface)
4. [High-level execution flow](#4-high-level-execution-flow)
5. [Step 1 — Startup and thread initialisation](#5-step-1--startup-and-thread-initialisation)
6. [Step 2 — Reading the input file list](#6-step-2--reading-the-input-file-list)
7. [Step 3 — The prefetch pipeline (cross-file parallelism)](#7-step-3--the-prefetch-pipeline-cross-file-parallelism)
8. [Step 4 — Processing each input file (outer loop)](#8-step-4--processing-each-input-file-outer-loop)
9. [Step 5 — Processing each DF folder (inner loop)](#9-step-5--processing-each-df-folder-inner-loop)
10. [Step 6 — The serial phase (first-occurrence trees)](#10-step-6--the-serial-phase-first-occurrence-trees)
11. [Step 7 — The parallel phase (deferred trees)](#11-step-7--the-parallel-phase-deferred-trees)
12. [Step 8 — `processTree` in detail](#12-step-8--processtree-in-detail)
13. [Step 9 — The max-size flush boundary](#13-step-9--the-max-size-flush-boundary)
14. [Step 10 — Finalisation and output](#14-step-10--finalisation-and-output)
15. [Thread-safety analysis](#15-thread-safety-analysis)
16. [Memory footprint analysis](#16-memory-footprint-analysis)
17. [Coherence with other AliceO2 threading systems](#17-coherence-with-other-aliceo2-threading-systems)
18. [Build system integration](#18-build-system-integration)
19. [Potential follow-up improvements](#19-potential-follow-up-improvements)
20. [Known bugs and post-mortems](#20-known-bugs-and-post-mortems)
    - [20.1 Heap corruption from concurrent `SetBranchAddress` on shared output trees](#201-heap-corruption-from-concurrent-setbranchaddress-on-shared-output-trees)
    - [20.2 Null `GetAddress()` segfault after fast CloneTree](#202-null-getaddress-segfault-after-fast-clonetree)

---

## 1. Background — what are AO2D files?

ALICE O2 produces analysis data in ROOT files whose internal structure follows a strict
convention. Each file contains:

- A `metaData` object (a `TMap`) with key-value pairs describing the production.
- A `parentFiles` object (a `TMap`) listing the raw data files this AOD was derived from.
- One or more **Data Frame (DF) directories**, named `DF_<timestamp>`, e.g. `DF_2310130`.

Each DF directory holds a set of **ROOT TTrees**, one per analysis table. Typical tables
include:

| Tree name       | Contents                                    |
|-----------------|---------------------------------------------|
| `O2collision`   | One row per reconstructed collision vertex  |
| `O2track`       | One row per reconstructed track             |
| `O2caloCell`    | Calorimeter cell hits                       |
| `O2fv0a`        | Forward V0-A detector hits                  |
| ... (20–50 trees per DF) | ...                              |

**Index columns** are the critical complication. A row in `O2track` does not store the
collision data inline — it stores an integer that is the *row number* in `O2collision`
that this track belongs to. These integer references (stored in branches whose names start
with `fIndex`) are valid only within a single DF: row 0 of `O2collision` in `DF_A` is a
different collision from row 0 of `O2collision` in `DF_B`. When DFs are merged, all these
row numbers must be shifted by the number of rows already accumulated in the output.

This is the core logic the merger must perform correctly.

---

## 2. What the merger must do

Given a list of input `.root` files, the merger must produce a single output `.root` file
that:

1. Combines all DF directories from all input files into a smaller number of output DF
   directories (bounded by `--max-size` to control output directory sizes).
2. Rewrites every `fIndex*` branch value so that row references remain valid across the
   merged tables.
3. Preserves metadata (taking the first file's version as authoritative and warning on
   differences).
4. Preserves the union of all `parentFiles` entries.
5. Does all of this **as fast as possible** using available CPU cores.

---

## 3. Command-line interface

```
o2-aod-merger [options]

  --input  <file.txt>     Path to a text file listing input .root files, one per line.
                          Default: input.txt
  --output <file.root>    Output ROOT file name. Default: AO2D.root
  --max-size <bytes>      Maximum uncompressed size of a single output DF directory.
                          Default: 100 000 000 (100 MB). Set to 0 to never split DFs.
  --skip-non-existing-files
                          If set, missing files are skipped with a warning rather than
                          causing a fatal error.
  --skip-parent-files-list
                          If set, the parentFiles map is not written to the output.
  --compression <id>      ROOT compression algorithm+level identifier. Default: 505
                          (LZ4 level 5). Format: algorithm*100 + level.
    --merge-by-name         Only merge TTrees from folders with the same name.
                                                    When enabled, accumulated output trees are flushed as soon
                                                    as the input DF folder name changes.
  --workers <N>           Number of parallel worker threads. Default: all logical cores
                          reported by std::thread::hardware_concurrency().
  --verbosity <level>     0 = silent, 1 = per-folder, 2 = per-tree, 3 = per-branch.
                          Default: 2.
```

---

## 4. High-level execution flow

The diagram below shows every phase of execution. Serial operations appear on the main
thread (left column). Parallel activity is shown branching to the right.

```
MAIN THREAD                                   OTHER THREADS
─────────────────────────────────────────     ──────────────────────────────────────
Parse arguments, initialise ROOT MT,
create output TFile, read input list
          │
          │  std::async(launch::async) ──────► OS thread: TFile::Open(inputFiles[0])
          │
          ▼
  ┌─ for fi = 0 .. N-1: ──────────────────────────────────────────────────────────┐
  │                                                                                │
  │  prefetchFuture.get()                                                          │
  │  (waits for the open launched in the previous iteration)                      │
  │                                                                                │
  │  std::async(launch::async) ──────────────► OS thread: TFile::Open(inputFiles[fi+1])
  │  (immediately overlaps with CPU work below)                                   │
  │                                                                                │
  │  for each DF_X in inputFiles[fi]:                                             │
  │  │                                                                            │
  │  │  SERIAL PHASE: first-occurrence trees                                      │
  │  │  ─ mkdir DF_X in output (if new)                                           │
  │  │  ─ CloneTree + processTree(alreadyCopied=true)                             │
  │  │  ─ record which trees were deferred                                        │
  │  │                                                                            │
  │  │  PARALLEL PHASE: deferred trees                                            │
  │  │  arena.execute → tbb::task_group:                                          │
  │  │    ├─ task 0: open file[fi], fill O2track ──────────────► TBB worker 0    │
  │  │    ├─ task 1: open file[fi], fill O2collision ───────────► TBB worker 1   │
  │  │    ├─ task 2: open file[fi], fill O2caloCell ────────────► TBB worker 2   │
  │  │    └─ ...                                                                  │
  │  │  tg.wait()  ◄── main thread blocks here                                   │
  │  │                                                                            │
  │  │  Serial aggregation: collect newMinOffset, exitCode                        │
  │  │                                                                            │
  │  │  Update offsets[], check max-size, flush if needed                        │
  │  │                                                                            │
  │  └─ (next DF folder)                                                         │
  │                                                                                │
  │  inputFile->Close(); delete inputFile                                         │
  └────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
  Write remaining trees, write output TFile, print statistics
```

---

## 5. Step 1 — Startup and thread initialisation

```cpp
unsigned short int actualNWorkers =
    (nWorkers > 0) ? nWorkers
                   : static_cast<int>(std::thread::hardware_concurrency());

ROOT::EnableImplicitMT(actualNWorkers);
tbb::task_arena arena(actualNWorkers);
```

### `ROOT::EnableImplicitMT`

ROOT's internal I/O layer is not thread-safe by default: if two threads call
`TTree::Fill()` on trees that live inside the same `TFile`, they would corrupt each
other's internal basket buffers. `ROOT::EnableImplicitMT(N)` activates ROOT's
TBB-backed serialisation layer, which serialises basket-level writes to the output
`TFile` using internal locks. This makes it safe for multiple threads to call `Fill()` on
different trees concurrently — ROOT will queue the writes correctly.

The `N` passed here sets the size of ROOT's internal TBB thread pool. It is important
that this value matches `actualNWorkers` so ROOT does not spin up more threads than we
intend to use.

### `tbb::task_arena`

An **arena** is a bounded worker pool in Intel TBB. By default TBB has a global uncapped
pool; the arena restricts how many of those workers our code can use at once. This
prevents the merger from accidentally interfering with ROOT's own internal TBB usage and
gives us a clean boundary on CPU consumption.

`arena.execute([&]() { ... })` runs the lambda inside the arena — any TBB work spawned
inside will use at most `actualNWorkers` threads.

### `std::atomic<long> currentDirSize{0}`

This counter tracks the total uncompressed bytes written to the current output DF
directory. It is declared as `std::atomic<long>` so that TBB tasks can increment it from
multiple threads simultaneously without data races. See §11 and §13 for its role.

---

## 6. Step 2 — Reading the input file list

```cpp
std::vector<TString> inputFiles;
{
    std::ifstream in;
    in.open(inputCollection);
    TString pathBuf;
    while (in >> pathBuf) {
        if (!pathBuf.IsNull()) {
            inputFiles.push_back(std::move(pathBuf));
        }
    }
}

for (const auto& p : inputFiles) {
    if (p.BeginsWith("alien:") && !gGrid) {
        printf("Connecting to AliEn...\n");
        TGrid::Connect("alien:");
        break;
    }
}
```

All paths are read into a `std::vector` **before the main loop begins**. This serves two
purposes:

1. **Indexing**: the prefetch pipeline (§7) needs to know `inputFiles[fi+1]` while
   processing `inputFiles[fi]`. A streaming one-line-at-a-time reader cannot look ahead.

2. **AliEn safety**: files hosted on the CERN AliEn grid are accessed via XRootD URLs
   starting with `alien:`. Connecting to the grid requires calling `TGrid::Connect()`,
   which initialises the global `gGrid` object. This initialisation is not thread-safe and
   must happen on the main thread **before** any background thread calls `TFile::Open`.
   The pre-scan loop guarantees this.

---

## 7. Step 3 — The prefetch pipeline (cross-file parallelism)

Opening a file — especially over the network (XRootD/AliEn) — takes tens to hundreds of
milliseconds: a TLS handshake, file-catalogue lookup, and initial metadata read all happen
before any physics data is transferred. The CPU is completely idle during this time.

The prefetch pipeline eliminates this idle time by overlapping the open of file `N+1` with
the processing of file `N`:

```
Timeline without prefetch:
  open[0] ── process[0] ── open[1] ── process[1] ── open[2] ── process[2] ...
  ──────────────────────────────────────────────────────────► time

Timeline with prefetch:
  open[0]
  ────────── process[0]
             open[1]
             ──────────── process[1]
                          open[2]
                          ──────────── process[2] ...
  ──────────────────────────────────────────────────────────► time
```

The implementation uses `std::future<TFile*>` and `std::async`:

```cpp
// Before the loop: kick off the very first file
std::future<TFile*> prefetchFuture;
if (!inputFiles.empty()) {
    const TString first = inputFiles[0];
    prefetchFuture = std::async(std::launch::async, [first]() -> TFile* {
        return TFile::Open(first, "READ");
    });
}

for (size_t fi = 0; fi < inputFiles.size() && exitCode == 0; ++fi) {
    const TString currentFilePath = inputFiles[fi];

    // 1. Retrieve this iteration's file (may block if not ready yet)
    TFile* inputFile = prefetchFuture.get();

    // 2. Immediately request the next file so its open overlaps our CPU work
    if (fi + 1 < inputFiles.size()) {
        const TString nextPath = inputFiles[fi + 1];
        prefetchFuture = std::async(std::launch::async, [nextPath]() -> TFile* {
            return TFile::Open(nextPath, "READ");
        });
    }

    // 3. Process the current file (CPU-bound; next open runs in background)
    ...
    inputFile->Close();
    delete inputFile;
}
```

### Why `std::async` and not a TBB task?

`TFile::Open` is a **blocking I/O operation**. If it were submitted as a TBB task, it
would occupy a TBB worker thread for the entire duration of the open — that worker would
sleep on a network socket, unable to do any CPU work. With `std::async(launch::async)`,
the C++ runtime spawns a dedicated OS thread whose only job is to make the blocking call
and go to sleep until the OS wakes it up. The TBB pool is completely unaffected.

### Why capture `nextPath` by value in the lambda?

The lambda `[nextPath]() -> TFile* { return TFile::Open(nextPath, "READ"); }` captures
the path **by value**. If it captured by reference, `nextPath` would refer to a local
variable that goes out of scope at the end of the loop iteration — the background thread
would then read freed memory (undefined behaviour). Capturing by value makes a private
copy of the string that lives as long as the lambda itself.

---

## 8. Step 4 — Processing each input file (outer loop)

```cpp
TList* keyList = inputFile->GetListOfKeys();
keyList->Sort();
```

The key list is sorted alphabetically. This ensures that `DF_` directories are visited in
a deterministic order (their names embed a timestamp integer). Metadata keys (`metaData`,
`parentFiles`) appear before `DF_*` keys alphabetically and are handled by name checks at
the top of the key loop.

### metaData handling

The first file's `TMap` is written to the output immediately. For subsequent files, each
key-value pair is compared against the stored version and a warning is printed if they
differ. This is non-fatal because in practice the metadata values are nearly identical
across files from the same production.

### parentFiles handling

The `parentFiles` map accumulates entries from all input files into a single `TMap` that
is written to the output at the very end (§14). Entries from different files are simply
merged with no deduplication required because each parent file path is unique.

---

## 9. Step 5 — Processing each DF folder (inner loop)

```cpp
for (auto key1 : *keyList) {
    if (!((TObjString*)key1)->GetString().BeginsWith("DF_")) {
        continue;  // skip metaData, parentFiles
    }
    auto dfName = ((TObjString*)key1)->GetString().Data();
    auto folder = (TDirectoryFile*)inputFile->Get(dfName);
    auto treeList = folder->GetListOfKeys();
    treeList->Sort();
    ...
}
```

For each DF folder:

1. The list of tree names is sorted for determinism and duplicate keys are removed (ROOT
   can store multiple cycles of the same key; only the most-recent cycle is relevant).
2. The code walks over the sorted tree list and classifies each tree as either
   **first-occurrence** (serial path) or **already-existing** (parallel path).

---

## 10. Step 6 — The serial phase (first-occurrence trees)

The first time the merger encounters a tree named `O2track` it does not yet have an output
tree to append to. It must *create* one. This is inherently serial because:

- `outputFile->mkdir(dfName)` modifies the output file's directory structure.
- `inputTree->CloneTree(...)` creates a new `TTree` object registered in the output file.
- Both operations touch global ROOT I/O state.

```cpp
outputDir->cd();
auto outputTree = inputTree->CloneTree(-1, (fastCopy) ? "fast" : "");
currentDirSize += inputTree->GetTotBytes();
outputTree->SetAutoFlush(0);
trees[treeName] = outputTree;
```

**`CloneTree(-1, "fast")`**: Creates a new output tree with the same branch structure and
immediately copies all entries from the input tree. The `"fast"` flag tells ROOT to copy
compressed basket data directly without decompression/recompression — much faster for
large trees. For small trees (< 10 MB uncompressed) the fast path can produce
under-filled baskets, so it is skipped.

**`SetAutoFlush(0)`**: By default ROOT flushes basket buffers to disk when they reach a
configurable threshold. Setting it to `0` disables automatic flushing — all basket data
accumulates in memory until the output directory is explicitly flushed at the max-size
boundary (§13). This is intentional: it lets ROOT optimise basket fill levels, at the
cost of holding more data in RAM.

**`processTree(alreadyCopied=true)`**: Even though `CloneTree` already copied the data,
`processTree` is still called once with `alreadyCopied=true`. In this mode no `Fill()` is
issued — the function only walks the index columns to find the minimum unassigned index
value (§12). This initialises `unassignedIndexOffset[treeName]` for the merging of
subsequent files.

---

## 11. Step 7 — The parallel phase (deferred trees)

For trees that **already exist** in the output (i.e., this is the 2nd, 3rd, ... input
file), the work is:

- Read entries from the input tree.
- Adjust index column values.
- Append to the output tree with `Fill()`.

These operations are **independent across trees**: the data for `O2track` and `O2collision`
can be appended simultaneously because they write to different output `TTree` objects.
This independence is exploited by the parallel phase.

### Overview of the parallel machinery

```
main thread                              TBB worker threads
──────────────────                       ──────────────────────────────
arena.execute([&]{
  tbb::task_group tg;

  tg.run( task_0 ) ───────────────────► fill O2track
  tg.run( task_1 ) ───────────────────► fill O2collision
  tg.run( task_2 ) ───────────────────► fill O2caloCell
  ...

  tg.wait()       ◄──────────────────── (all tasks done)
})
```

### The task lambda in full

```cpp
arena.execute([&]() {
    tbb::task_group tg;
    for (size_t ti = 0; ti < deferredTrees.size(); ++ti) {
        tg.run([&, ti, currentFilePath]() {
            const auto& dt = deferredTrees[ti];
            int startUnassigned = unassignedIndexOffset.count(dt.treeName)
                                      ? unassignedIndexOffset.at(dt.treeName) : 0;

            // Open a private TFile handle for this task
            auto privateFile = TFile::Open(currentFilePath, "READ");
            if (!privateFile || privateFile->IsZombie()) {
                taskResults[ti] = {startUnassigned, 5};
                delete privateFile;
                return;
            }

            // Get a private TTree pointer out of the private file
            auto privateTree = static_cast<TTree*>(
                privateFile->Get(Form("%s/%s", dfName, dt.treeName.c_str())));

            // Do the actual entry-level merge work
            auto [bytes, newMinOffset, tCode] = processTree(
                privateTree, trees.at(dt.treeName),
                /*alreadyCopied=*/false, dt.fastCopy,
                snapshotOffsets, startUnassigned);

            // Accumulate bytes into the shared atomic counter (no lock needed)
            if (bytes > 0) {
                currentDirSize.fetch_add(bytes, std::memory_order_relaxed);
            }

            taskResults[ti] = {newMinOffset, tCode};
            privateFile->Close();
            delete privateFile;
        });
    }
    tg.wait();
});
```

### Why each task must open its own `TFile`

ROOT's `TFile` is **not thread-safe**. Internally, each `TFile` maintains a single I/O
buffer (`TBuffer`) that is reused across all reads. If two threads called
`someTree->GetEntry(i)` on trees from the same `TFile`, they would overwrite each other's
buffer mid-read, producing silent data corruption.

The solution is straightforward: each task opens its own `TFile` handle pointing to the
same file on disk. The operating system and the storage system handle concurrent reads from
the same file perfectly fine — only the in-process ROOT objects need to be isolated.

```
disk file: /data/AO2D_2310130.root
    │
    ├──► TFile* privateFile_task0   (task 0's private handle)
    ├──► TFile* privateFile_task1   (task 1's private handle)
    ├──► TFile* privateFile_task2   (task 2's private handle)
    └──► TFile* inputFile           (main thread's original handle, used for CloneTree)
```

Each private handle is opened, used, closed, and deleted within its own task. No
cross-task sharing.

### Why `currentFilePath` is captured by value

The task lambda captures `[&, ti, currentFilePath]`:
- `&` — captures everything else by reference. The referenced objects (`deferredTrees`,
  `snapshotOffsets`, `trees`, `taskResults`, `dfName`) are not modified during the
  parallel phase, so concurrent reads are safe.
- `ti` — captured by value. `ti` is the loop counter on the main thread. By the time a
  task runs on a worker thread, the main thread's `ti` may have already advanced to a
  different value. Capturing by value freezes the correct index for each task.
- `currentFilePath` — captured by value for the same reason. The file path for this
  iteration of the outer `for` loop must not change even if the outer loop were to
  advance (in this code it cannot, because of `tg.wait()`, but the capture-by-value is
  still the correct and clear way to express it).

### `snapshotOffsets` — immutable view during the parallel phase

```cpp
const std::map<std::string, int> snapshotOffsets = offsets;
```

The `offsets` map maps tree names to cumulative row counts. It must be read by every
task to compute index shifts. After the parallel phase, `offsets` is updated again on the
main thread. Snapshotting it into a `const` copy before launching tasks means tasks
always read a fixed, consistent view of offsets — no locking required.

### `std::atomic<long> currentDirSize` and `memory_order_relaxed`

```cpp
currentDirSize.fetch_add(bytes, std::memory_order_relaxed);
```

`currentDirSize` tracks the total uncompressed bytes written to the current output DF
directory. Multiple tasks write to it concurrently, so it must be atomic.

`memory_order_relaxed` is the weakest memory ordering: it guarantees the operation is
atomic (no partial update, no torn read) but makes no promises about the ordering of
this store relative to other memory operations. This is correct here because:
- The final value of `currentDirSize` is only read by the main thread **after**
  `tg.wait()`, which provides a full synchronisation barrier (acquire/release semantics).
- The only thing that matters is that the total sum is correct, not the order in which
  individual tasks contributed.

Using `relaxed` instead of `seq_cst` or `acq_rel` allows the compiler and CPU to optimise
the atomic add more aggressively.

### `TaskResult` — per-task output

Each task writes its results into a pre-allocated slot:

```cpp
struct TaskResult {
    int newMinOffset;  // smallest negative index seen (for unassigned indices)
    int exitCode;      // 0 = success, >0 = error
};
std::vector<TaskResult> taskResults(deferredTrees.size());
```

Slot `taskResults[ti]` is written exclusively by task `ti` and read by the main thread
only after `tg.wait()`. No two tasks ever access the same slot. No lock is needed.

Bytes are not stored here — they go directly to the atomic counter.

### After `tg.wait()` — serial aggregation

```cpp
for (size_t ti = 0; ti < deferredTrees.size(); ++ti) {
    unassignedIndexOffset[deferredTrees[ti].treeName] = taskResults[ti].newMinOffset;
    if (taskResults[ti].exitCode > 0 && exitCode == 0) {
        exitCode = taskResults[ti].exitCode;
    }
}
```

The main thread collects the minimum unassigned index (needed for the next file's
processing) and propagates any error code. This is fully serial and requires no
synchronisation beyond `tg.wait()`.

---

## 12. Step 8 — `processTree` in detail

`processTree` is a lambda defined once at the beginning of `main` and called from both
the serial phase (via `alreadyCopied=true`) and from parallel tasks:

```cpp
auto processTree = [verbosity](
    TTree* inputTree,
    TTree* outputTree,
    bool   alreadyCopied,
    bool   fastCopy,
    const std::map<std::string, int>& snapshotOffsets,
    int    startUnassignedOffset
) -> std::tuple<long long, int, int>;
//              ^^^^^^^^   ^^^  ^^^
//              bytesAdded  newMinUnassignedOffset  exitCode
```

### Private per-task buffer allocation

`CopyAddresses` is **not used**. The previous approach (`CopyAddresses` + `GetAddress()
to recover the output buffer pointer) was found to produce null pointers after a fast
`CloneTree`, crashing in `*(outBuf) = val` (see §20.2). The current approach allocates a
private typed buffer for every branch in the input tree and connects **both** the input
and the output tree to that buffer:

```cpp
inputTree->SetBranchAddress(branchName, buffer);
outputTree->SetBranchAddress(branchName, buffer);
```

`GetEntry(ei)` then fills `buffer` from the input file, any index adjustment is made
in-place in `buffer`, and `Fill()` reads the final value out of `buffer` — a single
allocation that serves as both read and write location. The `outputTree` is exclusive to
one `treeName`, so calling `SetBranchAddress` on it from the task body is safe (no two
tasks can share the same `outputTree`).

**Thread safety note**: the original code used `static TClass* cls` to cache a class
pointer for VLA branch type lookup. A `static` local variable is initialised once and
shared across all calls — a data race if called from multiple threads simultaneously. It
is changed to a plain local variable:

```cpp
TClass* cls = nullptr;  // not static — each task gets its own copy, no race
```

### Branch classification and buffer allocation

Every branch in the input tree falls into one of four categories:

| Category | Detection | Buffer type |
|----------|-----------|-------------|
| VLA (variable-length array) | leaf has a `GetLeafCount()` parent | `char[maximum × typeSize]` |
| `fIndexArray*` | VLA branch whose name begins `fIndexArray` | same VLA buffer; also enters `indexList` |
| `fIndexSlice*` | fixed-size `int[2]` (start, length) | `int[2]` |
| `fIndex*` (plain) | name begins `fIndex`, does not end `_size` | `int` |
| scalar / fixed-size array | everything else | `char[leaf->GetLen() × leaf->GetLenType()]` |

For scalar branches `GetLen()` returns 1; for fixed-size array branches (e.g., `fCovMat[15]`) it returns the array length. Using `leaf->GetLen() * leaf->GetLenType()` rather than just the per-element type size is critical: allocating only one element's worth of bytes for a fixed-size array branch causes ROOT to overflow the buffer during `GetEntry()`, corrupting the heap.

For every index-containing branch (`fIndexArray`, `fIndexSlice`, `fIndex`), an
`IndexEntry` is pushed onto `indexList`:

```cpp
struct IndexEntry {
    int* buf;    // pointer into the private buffer; adjusted in-place
    int  offset; // cumulative row count to add
};
```

`snapshotOffsets[targetTable]` is the number of rows already written to that table in the
output — adding it to the raw index value translates a per-DF local row number into a
global merged row number.

For scalar non-index branches the buffer is allocated using `leaf->GetLen() * leaf->GetLenType()`. `GetLen()` returns 1 for true scalars and N for fixed-size array branches (`float fCovMat[15]` → `GetLen()=15`). A defensive minimum of `sizeof(Long64_t)` is used if the product is zero, which can occur for rare object-typed branches that should not appear in flat AOD trees.

### The entry loop

```cpp
for (Long64_t ei = 0; ei < entries; ei++) {
    for (auto& idx : indexList) {
        *(idx.buf) = 0;  // reset buffer before read
    }
    inputTree->GetEntry(ei);

    for (auto& idx : indexList) {
        int val = *(idx.buf);
        if (val < 0) {
            // "unassigned" index — kept as a large negative unique ID
            val += startUnassignedOffset;
            newMinOffset = std::min(newMinOffset, val);
        } else {
            val += idx.offset;  // shift by cumulative row count
        }
        *(idx.buf) = val;  // write back in-place; outputTree->Fill() reads from here
    }

    if (!alreadyCopied) {
        int nbytes = outputTree->Fill();
        if (nbytes > 0) bytesAdded += nbytes;
    }
}
```

For each entry:
1. Reset all index buffers (necessary because VLA branches may not write all slots).
2. Read the entry from the input — this fills all branch buffers.
3. Walk every registered index buffer and add the offset.
4. If this is not the `alreadyCopied` path, write the modified entry to the output tree.

### Unassigned indices

In O2 analysis, an index value of `-1` (or any large negative number) means "this row
does not reference any row in the target table". When merging, each DF block's unassigned
indices must remain distinct. The `startUnassignedOffset` mechanism ensures this:

- For the first DF in the output, `startUnassignedOffset = 0`, so `-1 + 0 = -1`.
- For the second DF, `startUnassignedOffset` is set to the minimum unassigned value seen
  in the first DF minus 1, so the second block's unassigned values stay below any value
  from the first block.

This guarantees that unassigned indices from different input blocks never collide.

### Fast copy path (no index branches)

```cpp
} else if (!alreadyCopied) {
    Long64_t nbytes = outputTree->CopyEntries(inputTree, -1, fastCopy ? "fast" : "");
    if (nbytes > 0) bytesAdded += nbytes;
}
```

If a tree has no index branches at all (e.g., a table of calibration constants with no
foreign-key references), no per-entry manipulation is needed. `CopyEntries("fast")` copies
compressed basket data directly, bypassing decompression entirely. This is significantly
faster for large trees and is used when `inputTree->GetTotBytes() > 10 MB`.

### Cleanup

All index buffers allocated with `new` are freed before the function returns. No memory
is leaked even if the branch-scan loop allocated many buffers.

---

## 13. Step 9 — Flush boundaries (`--max-size` and `--merge-by-name`)

```cpp
auto flushTrees = [&](bool resetState) {
    if (!outputDir) {
        return;
    }
    for (auto const& tree : trees) {
        outputDir->cd();
        tree.second->Write();
        sizeCompressed[tree.first] += tree.second->GetZipBytes();
        sizeUncompressed[tree.first] += tree.second->GetTotBytes();
        delete tree.second;
    }
    if (resetState) {
        outputDir = nullptr;
        trees.clear();
        offsets.clear();
        mergedDFs = 0;
        currentDirSize = 0;
    }
};
```

The merger uses one shared flush helper for both boundary conditions:

1. **Size boundary**: when `maxDirSize == 0 || currentDirSize > maxDirSize`, the current
   output directory is flushed and state is reset (`flushTrees(true)`).
2. **Name boundary** (`--merge-by-name`): if the incoming folder name differs from the
   currently open output directory name, the merger flushes and resets immediately,
   ensuring only same-name DF folders are merged together.

This keeps the original batching behavior while adding a strict structure-preserving mode
needed for workflows like MC-DATA embedding.

### Memory implication of `SetAutoFlush(0)`

With `SetAutoFlush(0)`, all basket data for the current output DF accumulates in RAM until
the flush point. With `--max-size 100000000` (the default), this caps the uncompressed
in-memory accumulation at ~100 MB. For standard O2 production files this is well within
budget. If you are processing very large custom DFs without a max-size cap
(`--max-size 0`), be aware that all data accumulates in memory until the end of the file.

---

## 14. Step 10 — Finalisation and output

After the outer file loop ends, the remaining open output DF (if any) is flushed:

```cpp
flushTrees(false);
outputFile->Write();
outputFile->Close();
```

`outputFile->Write()` writes the top-level file header, the key index, and the
`metaData` / `parentFiles` objects. `outputFile->Close()` flushes all pending I/O and
releases the file handle.

If `exitCode != 0`, the output file is deleted with `gSystem->Unlink(...)` so that
downstream jobs do not accidentally use an incomplete or corrupt output.

On success the merger prints a per-tree size summary showing compressed and uncompressed
byte counts as fractions of the total output, which is useful for diagnosing compression
efficiency.

---

## 15. Thread-safety analysis

This section summarises every shared data structure that could be a source of data races
and explains why each is safe.

| Object | Accessed from | Safety mechanism |
|--------|---------------|-----------------|
| Output `TFile` basket writes | Multiple TBB tasks calling `outputTree->Fill()` | `ROOT::EnableImplicitMT` serialises basket writes internally |
| Output `TTree` branch addresses | TBB tasks call `SetBranchAddress` on the output tree during task setup | Safe because each `outputTree` is exclusive to exactly one `treeName` — no two parallel tasks ever operate on the same `outputTree` object. `ROOT::EnableImplicitMT` does not cover this, but exclusive ownership does. |
| `trees` map (read: `trees.at(name)`) | TBB tasks (read-only) | Written only in serial phase, fully constructed before parallel phase; const read is safe |
| `snapshotOffsets` map | TBB tasks (read-only) | `const` copy taken before parallel phase |
| `deferredTrees` vector | TBB tasks (read-only) | Written only in serial phase before parallel phase |
| `taskResults[ti]` | Task `ti` writes, main thread reads after `tg.wait()` | Non-overlapping slot assignment + `tg.wait()` barrier |
| `currentDirSize` | TBB tasks (writes) + main thread (read after barrier) | `std::atomic<long>` with `memory_order_relaxed` + `tg.wait()` acquire |
| `unassignedIndexOffset` map | Main thread only (before and after parallel phase) | Never accessed from tasks |
| `offsets` map | Main thread only | Never accessed from tasks (tasks use `snapshotOffsets`) |
| `inputFile` (shared file handle) | Main thread only | Tasks open their own private handles; `inputFile` is not passed to tasks |
| `dfName` (pointer into ROOT's key string) | Read by tasks via `Form(...)` | Read-only; string is stable while `inputFile` is open |
| `privateFile` / `privateTree` | Per-task local variables | No sharing |

---

## 16. Memory footprint analysis

On a node with 8 cores and 2 GB/core (16 GB total):

| Component | Estimated size |
|-----------|---------------|
| ROOT framework baseline (TClass registry, etc.) | ~300–500 MB |
| Output DF basket accumulation (`SetAutoFlush(0)`, bounded by `--max-size`) | ~100 MB |
| `ROOT::EnableImplicitMT(8)` compression thread-pool buffers | ~50–150 MB |
| Per-task private `TFile` + ROOT I/O buffer (8 tasks × ~5–30 MB) | ~40–240 MB |
| Per-task `CopyEntries("fast")` basket transfer buffer | ~10–50 MB per task |
| Prefetch `TFile` (next file, already open) | ~5–30 MB |
| VLA/index branch buffers per task | KB range, negligible |
| **Total typical** | **~1–1.5 GB** |
| **Total worst case** | **~2.5–3 GB** |

16 GB provides comfortable headroom. The only scenario that can approach OOM is
processing input files that were themselves produced without a max-size limit (single DFs
of several GB), because `SetAutoFlush(0)` accumulates all their data in memory before
writing.

---

## 17. Coherence with other AliceO2 threading systems

Three independent threading systems are present in `~/Git/AliceO2`.

### GPU Tracking — `GPU/GPUTracking/Base/`

The most complete and recent TBB integration in the project. It uses
`tbb::task_arena`, `tbb::parallel_for`, and `tbb::global_control` across reconstruction
kernels.

| Aspect | GPU Tracking | AOD Merger |
|--------|-------------|------------|
| TBB include | `<oneapi/tbb.h>` (unified) | `<tbb/task_arena.h>`, `<tbb/task_group.h>` (split) |
| Arena | `tbb::task_arena` in `GPUReconstructionThreading` struct | `tbb::task_arena` local to `main` |
| Task dispatch | `tbb::parallel_for` inside `arena.execute` | `tbb::task_group` inside `arena.execute` |
| Arena isolation | `tbb::this_task_arena::isolate` (nested arenas) | Not used (single level) |
| Thread-count control | `tbb::global_control` + `TBB_NUM_THREADS` / `OMP_NUM_THREADS` | `--workers` flag or `hardware_concurrency()` |
| ROOT MT | Not used | `ROOT::EnableImplicitMT` for basket writes |
| Blocking I/O | Not applicable | `std::async(launch::async)` dedicated OS thread |

**Divergence explanations:**

- **Split headers vs `<oneapi/tbb.h>`**: The split headers (`<tbb/task_arena.h>` etc.)
  are the older TBB 2019 API. They map to the same symbols as the unified header but are
  a different include style. This is a cosmetic difference only.

- **No `tbb::global_control`**: GPU tracking creates this to cap the global TBB pool when
  multiple reconstruction instances coexist. The AOD merger is a standalone single-process
  executable; the arena alone is sufficient.

- **`task_group` vs `parallel_for`**: `parallel_for` with a `blocked_range` is optimal
  when all items are similar in cost and there are many of them (thousands). With 20–50
  trees per DF and highly variable per-tree cost (size varies by orders of magnitude),
  `task_group` is simpler and equally effective.

- **`std::async` for I/O**: GPU tracking never needs to overlap blocking I/O with CPU
  work (GPU memory copies use async CUDA streams). The AOD merger does, and an OS thread
  via `std::async` is the correct tool for blocking I/O overlap — not a TBB worker.

### TPC SpaceCharge — `Detectors/TPC/spacecharge/`

Uses `ROOT::EnableImplicitMT` + `ROOT::RDataFrame` for parallel I/O of 3D electric field
maps.

| Aspect | TPC SpaceCharge | AOD Merger |
|--------|----------------|------------|
| Parallelism API | `ROOT::RDataFrame` (implicit) | `tbb::task_group` (explicit) |
| `EnableImplicitMT` guard | Disable/re-enable if count changes | Called once at startup, fixed |
| Thread count | Per call-site | Set once via `--workers` |

The AOD merger calls `EnableImplicitMT` once before any `TFile` is opened, which is the
correct usage. The SpaceCharge's disable/re-enable pattern is needed because it may be
called from different contexts with different thread counts; the merger has no such
requirement.

### DPL Framework AsyncQueue — `Framework/Core/`

A lock-free inter-device message queue using `std::atomic` gates for scheduling.
Operates at a completely different level (inter-process data routing) with no overlap with
the merger. The merger's use of `std::atomic` is far simpler: a single counter with
`relaxed` ordering, fenced by `tg.wait()`.

---

## 18. Build system integration

```cmake
# Framework/AODMerger/CMakeLists.txt
o2_add_executable(merger
    COMPONENT_NAME aod
    SOURCES src/aodMerger.cxx
    PUBLIC_LINK_LIBRARIES ROOT::Core ROOT::Net TBB::tbb)
```

`TBB::tbb` is the canonical AliceO2 imported CMake target, defined in
`cmake/FindTBB.cmake`. It resolves to the correct include paths and library for the
system's TBB installation. `ROOT::Net` provides `TGrid`/XRootD support for AliEn files.

The project uses C++20 (`set(CMAKE_CXX_STANDARD 20)` in the root `CMakeLists.txt`).
The structured bindings `auto [bytes, newMinOffset, tCode] = processTree(...)` require
at least C++17.

---

## 19. Potential follow-up improvements

1. **Switch to `<oneapi/tbb.h>`** — the unified oneTBB 2021+ single header, consistent
   with GPU tracking.

2. **Add `tbb::global_control`** — allows the `TBB_NUM_THREADS` and `OMP_NUM_THREADS`
   environment variables to override `--workers`, matching the GPU tracking pattern:
   ```cpp
   const char* tbbEnv = getenv("TBB_NUM_THREADS");
   int n = (tbbEnv && atoi(tbbEnv) > 0) ? atoi(tbbEnv)
         : (nWorkers > 0)               ? nWorkers
         :                                tbb::info::default_concurrency();
   tbb::global_control gc(tbb::global_control::max_allowed_parallelism, n);
   ```

3. **`SetAutoFlush(-10000000LL)` guard for large DFs** — using `-10000000` instead of `0`
   tells ROOT to flush every ~10 MB of basket accumulation, capping per-DF RAM usage
   independent of `--max-size`. This matters when processing DFs that were produced
   without a size cap.

4. **Deep-buffer prefetch** — the current pipeline hides one file-open's latency.
   For very large files on slow storage, pre-reading and decompressing branch data for the
   next file into flat `std::vector<char>` buffers would hide decompression latency too,
   at the cost of more significant refactoring of `processTree`.

5. **Parallel DF processing within a file** — DFs in a file are independent for reading
   but sequential for writing (the `offsets` map is updated after each DF). A speculative
   read-ahead of subsequent DFs' branch data while writing the current DF would expose
   another level of parallelism, at the cost of additional memory and complexity.

---

## 20. Known bugs and post-mortems

### 20.1 Heap corruption from concurrent `SetBranchAddress` on shared output trees

**Symptom**

Launching the merger with `--workers 8` produced an immediate abort on the first input
file:

```
AOD merger started with:
  Input file: aodmerge_input.txt
  Output file name: AO2D_pre.root
  Parallel workers: 8
Processing input file: tf1/AO2D.root
malloc_consolidate(): unaligned fastbin chunk detected
Aborted (core dumped)
```

The merger ran correctly with `--workers 1` and with the previous single-threaded version.

**Root cause**

The original `processTree` began with:

```cpp
// WRONG — modifies the shared output tree from parallel tasks
outputTree->CopyAddresses(inputTree);
...
outputTree->SetBranchAddress(branchName, buffer);  // also wrong
```

With 8 TBB tasks running simultaneously, each task called `CopyAddresses` and
`SetBranchAddress` on its own output tree object. These calls walk ROOT's internal
`TObjArray` of branches and update raw pointers — non-atomic heap mutations. Although
each task operated on a *different* `TTree` object, all objects share ROOT's global
allocator (`malloc`), and the concurrent pointer writes from multiple threads produced
heap metadata corruption, triggering `malloc_consolidate`'s consistency check.

Note that `ROOT::EnableImplicitMT` only serialises **basket I/O writes** (i.e., calls
that happen inside `TTree::Fill` at the ROOT I/O layer). It does not protect arbitrary
mutations of `TTree` or `TBranch` objects themselves.

**The interim fix (later superseded — see §20.2)**

The direction of `CopyAddresses` was reversed and `SetBranchAddress` was called only on
the **private input tree**, never on the shared output tree:

```cpp
// Interim fix — correct direction but still used GetAddress()
inputTree->CopyAddresses(outputTree);  // reads output's buffers, installs them on input
inputTree->SetBranchAddress(branchName, readBuf);  // private tree only
// recover output branch's buffer address (read-only)
int* outBuf = reinterpret_cast<int*>(outputTree->GetBranch(branchName)->GetAddress());
```

This eliminated the heap corruption but introduced a new crash (§20.2) because
`GetAddress()` returns `nullptr` on trees created by `CloneTree(-1, "fast")`.

**Design rule derived from this bug**

> `CopyAddresses` must never be called on a `TTree` that may have null branch addresses.
> After a fast `CloneTree` the output tree's internal buffers are managed by ROOT and are
> not exposed via `GetAddress()`. Always allocate explicit private buffers and call
> `SetBranchAddress` on both trees directly.

---

### 20.2 Null `GetAddress()` segfault after fast CloneTree

**Symptom**

After applying the §20.1 interim fix, launching with `--workers 8` produced simultaneous
segmentation violations across all 7 active TBB worker threads:

```
 *** Break *** segmentation violation
...
#7  0x... in processTree::operator()() at aodMerger.cxx:263
```

All threads crashed at the same line: `*(idx.outBuf) = val`.

**Root cause**

`CloneTree(-1, "fast")` copies compressed basket data directly without ever setting
user-visible branch addresses. After the fast clone, `outputTree->GetBranch(name)->
GetAddress()` returns **null** — ROOT owns the basket memory internally and does not
expose it via the public `GetAddress()` API.

The interim fix stored this null pointer in `IndexEntry.outBuf`. The first time any task
attempted `*(idx.outBuf) = val`, it dereferenced null, crashing the thread. With 8
workers, all threads hit this path simultaneously.

**The final fix**

`GetAddress()` was removed from the data path entirely. The current implementation uses:

1. `inputTree->SetBranchAddress(...)` to bind all private per-task buffers.
2. `outputTree->CopyAddresses(inputTree)` once, after all input addresses are set.
3. `Fill()` reads through those copied addresses.

This keeps the output tree side free of per-branch pointer probing and avoids relying on
ROOT-internal address exposure after fast clone.

After each invocation, both trees are detached from the temporary buffers via
`ResetBranchAddresses()` (for output only when `alreadyCopied == false`) so no dangling
addresses survive into later calls.

**Design rule derived from this bug**

> Never call `TBranch::GetAddress()` on an output tree that was created by fast
> `CloneTree`. Bind private buffers on the input tree, then copy addresses to the output
> tree once per invocation, and reset branch addresses afterwards.

### 20.3 Allocator mismatch on `fIndexSlice` buffers (`malloc_consolidate` abort)

**Symptom**

The merger aborted very early with:

```
malloc_consolidate(): unaligned fastbin chunk detected
Aborted (core dumped)
```

typically while processing the first input file with multiple workers.

**Root cause**

`fIndexSlice` buffers are allocated as `new int[2]`, but they were temporarily tracked
inside a generic `char*` cleanup container and released through the wrong deletion path.
That allocator mismatch is undefined behavior and can corrupt malloc metadata, later
failing in `malloc_consolidate()`.

**Final fix**

`fIndexSlice` now has its own typed cleanup list (`std::vector<int*> slicePointers`) and
is always released with `delete[]` on `int*`. VLA byte buffers remain in the `char*`
container and are released with `delete[]` on `char*`.

**Design rule derived from this bug**

> Keep allocation/deallocation paths type-consistent. Do not mix typed arrays into
> generic byte-pointer cleanup lists unless ownership and deleter type are preserved.
