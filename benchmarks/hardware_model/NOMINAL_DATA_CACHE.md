# Source-local cache path scenarios

`nominal_data_cache.py` chooses a complete load path from source-local
addresses and explicit capacity hypotheses. It is called only after the
scheduler selects the next instruction issue. Exploring competing warps
never changes cache state.

Each four-byte local slot across a coherent warp maps to four complete
32-byte sectors. The source allocation determines frame extent; reused
physical slots retain the same addresses across waves, while trajectory
backing uses distinct addresses. L1 uses unified capacity minus shared
carveout. L2 uses an explicitly nominal equal per-SM capacity share, not a
claim of physical partition ownership. Both levels use fully associative
sector LRU; set conflicts and native line replacement remain unmodeled.

Loads choose one published complete path, 32 cycles for L1, an explicitly
selected 273 or 284.8 cycles for L2, and 571 cycles for DRAM. The maximum
sector-ready time supplies the warp result. These times are not summed.
Outstanding fills merge by sector and preserve readiness. A second request
cannot observe a cache hit before a fill completes. Fill-queue capacity is
explicitly unlimited as a capacity-ceiling approximation; no MSHR count or
queue penalty is invented.

Stores cover whole sectors and need no read for ownership in this layout
hypothesis. Their data enters the cache at the scenario's declared store
visibility. Write-through L1 and write-back are distinct named alternatives.
Dirty evictions and downstream writes are counted. Their bandwidth,
availability and drain delays remain unmodeled, including when a later load
revisits an evicted dirty sector. This is a capacity and load-path estimate,
not a proof of coherence timing or complete kernel latency.

All cache timestamps share one global SM clock. The engine runs every
synchronized wave, carries cache state, and sums actual wave costs. It does
not multiply a cold first wave. Cold state and explicit seed state are
separate hypotheses; seeded contents must fit their stated capacity. Warm
hits arising from completed previous waves require no free warmup phase.

Cache-enabled results contain `wave_schedules`; their `wave_cycles` field
is null because the waves need not have equal cost. `common_work.cycles`
and `common_work.cycles_per_warp_attempt` retain their comparison meaning.
Trace times are global and include the wave index. Exact store/load motif
contracts retain their measured joint service and update cache contents at
their declared visible time; individual load latency is not added again.

`nominal_scenarios.bind_data_cache` binds these hypotheses to the verified
policy allocation and source geometry. The standalone engine additionally
checks that cache frame extent matches the supplied typed plan. Independent
review is required after the external author checks at
`verification/nominal_data_cache_author_20260905/receipt.json`.
