# V-Engram

Simulation code for *V-Engram: Semantic Candidate Discovery over Kademlia for
Decentralized Vector Retrieval*.

The system maps a 1024-dimensional embedding to `L` sign-random-projection
sketches, treats each sketch as a Kademlia rendezvous target, and retrieves
candidates by walking the overlay toward those targets. This repository
reproduces every table and figure in the paper.

Release `v1.0-cc`, commit `f9f2663`.

## Layout

```
prepare/     corpus and embedding construction (two Colab notebooks plus
             a standalone PQ rebuild script; see prepare/README.md)
src/         routing, network and node models
data/        corpora, embeddings, ground truth, PQ codebooks (not in git)
run_*.sh     experiment drivers, one per group of results
analyze_*.py aggregation over seeds; each prints the numbers used in the paper
```

`main_simulation.py` is the discrete-event simulator: it walks the overlay,
counts messages and rounds, and models elapsed time. `main_simulation_v2.py`
resolves lookups by taking the globally closest peers instead of walking; it is
faster and is used for sweeps where routing is not the variable under study.
`main_churn_engine.py` models session churn and repair.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Data is not in the repository — `code100k_corpus_embeddings.npy` alone is
410 MB, above GitHub's per-file limit. Build it with `prepare/`, or request the
archive from the corresponding author and unpack into `data/`.

```bash
ls data/code_corpus_embeddings.npy data/code_ground_truth.json   # sanity check
```

## Reproducing the paper

Use the drivers. The bare defaults of `main_simulation.py` do not reproduce any
published result: they are development settings, `T=3` with 100 queries and a
ring routing table, chosen to make a smoke test finish quickly. Every published
figure comes from a driver that sets the flags explicitly.

Each driver skips work already on disk, so an interrupted run can be restarted
with the same command. All accept `PARALLEL=n` to bound concurrency.

| result | command | wall time |
|---|---|---|
| headline, termination and margin ablations, cost table | `PARALLEL=4 bash run_join_bootstrap.sh` | 12 h |
| projection independence, replication, static failure | `PARALLEL=4 bash run_two_sweeps.sh` | 2.5 h |
| replication at `r=5,10`; failure at `r=2,3` | `PARALLEL=4 bash run_sweeps_extra.sh` | 1 h |
| multi-probe Bucket-LSH baseline | `bash run_lsh_10seeds.sh` | 2 h |
| session churn and repair | `PARALLEL=4 bash run_churn.sh` | 4 h |
| weak scaling to 100k objects | `bash run_weakscaling.sh` | 3 h |

Then:

```bash
python3 analyze_join.py          # headline, termination, margin, cost
python3 analyze_two_sweeps.py    # independence, replication, failure
python3 analyze_lsh10.py         # Bucket-LSH baseline
python3 analyze_churn.py         # churn and repair
```

Every analysis script reports means with standard deviations and the seed count,
and paired comparisons carry a 95% confidence interval and an exact p-value.

### Running one configuration by hand

To reproduce the headline configuration for a single seed, every flag has to be
given; the defaults will not do it.

```bash
SKIP_PAYLOAD=1 ROUTING_TABLE=kbucket BOOTSTRAP=join \
  PQ_VARIANT=m512 SHARED_ORIGIN=1 NORMALIZE_ROWS=1 \
  python3 main_simulation.py --dataset code --nodes 10000 --seed 20235956 \
  --k-query 20 --multi-probe 8 --meta-anchors 1 --nq 500
```

This is what `run_join_bootstrap.sh` issues for each of the ten seeds.

## Seeds

Ten seeds are used throughout: `20235956, 1, 2, 3, 4, 5, 6, 7, 8, 9`. The first
is the primary seed and is the one used where a single run is reported. A seed
redraws peer identifiers and projection matrices; it does not change the corpus,
the query set, or the PQ codebook, all of which are fixed across seeds.

## Configuration flags

Behaviour is selected by environment variable rather than by editing code, so
alternatives can be run from one binary. Several defaults are development
settings rather than paper settings; the table says which is which.

| variable | values | default | paper | meaning |
|---|---|---|---|---|
| `ROUTING_TABLE` | `kbucket`, `ring` | `ring` | `kbucket` | `kbucket` is the Kademlia table the paper describes. `ring` is an earlier small-world construction, kept so the choice can be re-tested; the paper reports no results from it. |
| `BOOTSTRAP` | `join`, `oracle` | `join` | `join` | `join` has peers enter sequentially and learn contacts through real lookups. `oracle` hands each peer its globally nearest neighbours, which makes lookups converge perfectly; it is kept only to show that the resulting diagnostics are artefacts. |
| `PQ_VARIANT` | `m512`, `m256` | `m512` | `m512` | Product quantizer. `m256` is an earlier, lossier variant. |
| `STOP_RULE` | `stable`, `exhaust` | `stable` | both | Frontier-stability versus exhaustive top-K frontier termination. The paper compares them. |
| `FRONTIER_SCOPE` | `all`, `topk` | `all` | both | Whether the next peers to query come from the whole discovered candidate set or only the current frontier. The paper compares them. |
| `PROBE_ORDER` | `margin`, `random` | `margin` | both | Margin-ranked bit selection versus a random permutation of the same eligible positions. The paper compares them. |
| `SHARED_ORIGIN` | `1`, `0` | `1` | `1` | Whether all `L·T` probes of a query share one origin peer. `0` reproduces an earlier bug. |
| `NORMALIZE_ROWS` | `1`, `0` | `1` | `1` | Whether projection columns are L2-normalized before margins are compared. `0` reproduces an earlier bug. |
| `MEASURE_OVERLAP` | `1`, `0` | `0` | `1` for the termination table | Records XOR rank and inter-probe overlap. Expensive: it sorts the whole overlay per lookup. |
| `SKIP_PAYLOAD` | `1`, `0` | `0` | `1` except for the cost table | Skips payload retrieval, which the recall experiments do not need. |

Command-line defaults are development settings too: `--multi-probe 3` and
`--nq 100` against the paper's 8 and 500, and `--meta-anchors` unset against the
paper's 1.

## Known limitations of the model

Round-trip times are drawn independently from one distribution. There is no
bandwidth model, no serialization delay, no packet loss and no queueing, so
elapsed time is a lower bound and the paper treats it as auxiliary; the cost
metrics it argues from are critical-path rounds, RPCs, bytes and contacted
peers.

Record signatures and serialization are specified but not implemented — byte
counts are analytical, derived from field widths. The payload tier receives no
repair under churn. The churn study uses three seeds and is reported as an
exploratory diagnostic rather than as evidence that one maintenance policy
dominates another.

## Citation

Details will be added on acceptance.