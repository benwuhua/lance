# Wiki Schema

This wiki follows the [LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

## Directory Structure

```
docs/bench/wiki/
├── SCHEMA.md              # This file — wiki management rules
├── index.md               # Content-oriented catalog (all pages listed)
├── log.md                 # Chronological append-only record
├── entities/              # Things that exist (datasets, indexes, environments)
├── concepts/              # How things work (metrics, algorithms, architecture)
├── experiments/           # What we tried and what happened
└── roadmap/               # Where we're going (gap analysis, plans)
```

## Page Types

### Entity Pages (`entities/`)
Describe concrete objects with stable identity. Each page answers:
- What is it? (specs, parameters)
- Where is it? (paths, URIs)
- How was it built? (commands, scripts)
- Current status? (complete, deprecated, in-progress)

### Concept Pages (`concepts/`)
Explain technical ideas. Each page answers:
- What does this mean?
- Why does it matter for our benchmarks?
- How do we measure it?
- What are the trade-offs?

### Experiment Pages (`experiments/`)
Record what we did and what we learned. Each page has:
- `## Goal` — what we wanted to learn
- `## Setup` — how we ran it (commands, configs)
- `## Results` — raw data (tables, numbers)
- `## Findings` — interpreted conclusions
- `## Cross-refs` — links to related experiments, concepts, entities

### Roadmap Pages (`roadmap/`)
Future-oriented analysis. Each page has:
- `## Current state` — where we are now
- `## Gap analysis` — what's missing vs target
- `## Prioritized items` — P0/P1/P2 with rationale
- `## Status` — not started / in progress / done

## Operations

### Ingest
When new benchmark results arrive:
1. Read the result files
2. Update relevant entity pages (if new dataset/config)
3. Create or update experiment page with results
4. Update index.md
5. Append to log.md with format: `## [YYYY-MM-DD] ingest | <description>`

### Query
When answering questions about the project:
1. Read index.md first to find relevant pages
2. Drill into specific pages
3. Good answers get filed back as new pages or appended to existing ones

### Lint
Periodically check:
- Contradictions between pages
- Stale claims (mark with `[STALE]` prefix)
- Orphan pages (not linked from index.md)
- Missing cross-references
- Experiments without findings

## Conventions

- All pages are markdown with YAML frontmatter: `---\ntags: [tag1, tag2]\ndate: YYYY-MM-DD\n---`
- Cross-reference with relative links: `[see also](../concepts/metrics.md)`
- Data tables use markdown pipe tables
- Numbers always include units (ms, GB, %)
- Recall values to 4 decimal places
- Latency values to nearest ms
- Raw data stays on ECS (immutable), wiki has summaries
- One experiment per page, one concept per page
- Update index.md on every change

## File Naming

- Lowercase, hyphenated: `exact-rerank.md` not `ExactRerank.md`
- Entity pages: noun (dataset name, index type)
- Concept pages: noun phrase describing the concept
- Experiment pages: descriptive name of what was tested
- Roadmap pages: noun phrase describing the analysis

## Raw Sources (Immutable)

These are NOT part of the wiki. They live on ECS:
- Benchmark JSON results: `/data/work/tmp/pca512-results-v2/`, `/data/work/tmp/lance-benchmark-results/`
- Ground truth: `/data/work/tmp/gt_1b_shard0_top10k_5q_seed42.npz`
- Pareto results: `docs/bench/pareto-results.json`
- The wiki summarizes and interprets these, but never replaces them
