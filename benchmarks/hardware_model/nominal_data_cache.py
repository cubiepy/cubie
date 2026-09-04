"""Track conditional sector-cache paths on committed instruction issues."""

from collections import Counter, OrderedDict
from fractions import Fraction


def rational(value):
    """Read an exact cycle count from JSON or a Python rational."""
    if isinstance(value, dict):
        return Fraction(value["numerator"], value["denominator"])
    if isinstance(value, list):
        return Fraction(*value)
    return Fraction(value)


class NominalDataCache:
    """Model source-local sectors with explicit capacity and fill hypotheses.

    All times use one global SM clock across synchronized waves. Only
    committed instruction issues call this object. Pending reads merge
    without pretending that data is resident before its fill completes.
    """

    def __init__(self, specification, resident_warps):
        if specification.get("kind") != "sector_lru_data_cache":
            raise ValueError("Expected a sector LRU data-cache scenario")
        if not specification.get("provenance") or not specification.get(
                "assumption"):
            raise ValueError("Cache geometry requires a qualified hypothesis")
        if specification.get("outstanding_fills") != "unlimited_ceiling":
            raise ValueError("No measured finite fill-queue capacity supplied")
        if specification.get("write_policy") not in (
                "write_through_l1", "write_back"):
            raise ValueError("Explicit write policy is required")
        if specification.get("backing") not in (
                "reused_physical_slots", "trajectory_unique"):
            raise ValueError("Explicit local-frame backing is required")
        if specification.get("initial_state") not in (
                "cold", "explicit_seed"):
            raise ValueError("Explicit cold or seeded cache state required")
        self.specification = specification
        self.resident_warps = resident_warps
        self.frame = specification["frame_bytes_per_thread"]
        if type(self.frame) is not int or self.frame < 0 or self.frame % 4:
            raise ValueError("Local frame must be nonnegative aligned bytes")
        self.capacities = {}
        for level in ("l1", "l2"):
            size = specification[level + "_capacity_bytes"]
            if type(size) is not int or size < 0:
                raise ValueError("Cache capacity must be nonnegative bytes")
            self.capacities[level] = size // 32
        self.latencies = {key: rational(value) for key, value in
                          specification["load_path_cycles"].items()}
        if set(self.latencies) != {"l1", "l2", "dram"} or any(
                value <= 0 for value in self.latencies.values()):
            raise ValueError("Three positive complete load paths required")
        self.caches = {"l1": OrderedDict(), "l2": OrderedDict()}
        self.pending = {}
        self.counts = Counter()
        self.time = Fraction(0)
        if specification["initial_state"] == "explicit_seed":
            for level in ("l2", "l1"):
                sectors = specification.get(level + "_seed_sectors", [])
                if len(sectors) > self.capacities[level]:
                    raise ValueError("Explicit seed exceeds cache capacity")
                for sector in sectors:
                    if type(sector) is not int or sector < 0:
                        raise ValueError("Seed sector must be an address")
                    self.caches[level][sector] = False

    def sectors(self, detail, warp, wave):
        """Map full-warp local words to actual coalesced 32-byte sectors."""
        if detail["space"] != "local":
            raise ValueError("Data cache accepts thread-local frames only")
        low, size = detail["offset"], detail["bytes"]
        if low % 4 or size % 4 or low < 0 or low + size > self.frame:
            raise ValueError("Access exceeds the source-allocated local frame")
        slot = warp
        if self.specification["backing"] == "trajectory_unique":
            slot += wave * self.resident_warps
        # One 4-byte slot across 32 lanes occupies four 32-byte sectors.
        # The multiplication by 32 cancels in the sector address.
        return [slot * self.frame + word + sector
                for word in range(low, low + size, 4)
                for sector in range(4)]

    def insert(self, level, sector, dirty):
        """Insert a ready sector and count dirty downstream traffic."""
        cache = self.caches[level]
        capacity = self.capacities[level]
        if not capacity:
            if dirty:
                if level == "l1":
                    self.counts["l2_write_sectors"] += 1
                    self.insert("l2", sector, True)
                else:
                    self.counts["dram_write_sectors"] += 1
            return
        if sector in cache:
            dirty = cache.pop(sector) or dirty
        elif len(cache) == capacity:
            evicted, was_dirty = cache.popitem(last=False)
            self.counts[level + "_evictions"] += 1
            if was_dirty:
                self.counts[level + "_dirty_evictions"] += 1
                if level == "l1":
                    self.counts["l2_write_sectors"] += 1
                    self.insert("l2", evicted, True)
                else:
                    self.counts["dram_write_sectors"] += 1
        cache[sector] = dirty

    def advance(self, time):
        """Commit fills whose data is ready at the global issue timestamp."""
        time = rational(time)
        if time < self.time:
            raise ValueError("Cache time cannot run backwards across waves")
        self.time = time
        ready = sorted((record["ready"], key) for key, record
                       in self.pending.items() if record["ready"] <= time)
        for _, sector in ready:
            record = self.pending.pop(sector)
            if record["kind"] == "read":
                if record["fill_l2"]:
                    self.insert("l2", sector, False)
                self.insert("l1", sector, False)
            else:
                through = self.specification["write_policy"] == (
                    "write_through_l1")
                if through:
                    self.counts["l2_write_sectors"] += 1
                    self.insert("l2", sector, True)
                self.insert("l1", sector, not through)

    def load(self, detail, warp, wave, issue):
        """Return one full-path ready time per committed warp load."""
        self.advance(issue)
        issue = rational(issue)
        ready = issue
        paths = Counter()
        for sector in self.sectors(detail, warp, wave):
            self.counts["load_sectors"] += 1
            if sector in self.pending:
                record = self.pending[sector]
                completion = max(record["ready"], issue
                                 + self.latencies["l1"])
                path = "pending_merge"
            elif sector in self.caches["l1"]:
                dirty = self.caches["l1"].pop(sector)
                self.caches["l1"][sector] = dirty
                completion = issue + self.latencies["l1"]
                path = "l1"
            else:
                hit_l2 = sector in self.caches["l2"]
                if hit_l2:
                    dirty = self.caches["l2"].pop(sector)
                    self.caches["l2"][sector] = dirty
                path = "l2" if hit_l2 else "dram"
                completion = issue + self.latencies[path]
                self.pending[sector] = dict(
                    kind="read", ready=completion, fill_l2=not hit_l2,
                )
                if not hit_l2:
                    self.counts["dram_read_sectors"] += 1
            ready = max(ready, completion)
            paths[path] += 1
            self.counts["load_" + path + "_sectors"] += 1
        return dict(ready=ready, paths=dict(paths))

    def store(self, detail, warp, wave, issue, visible):
        """Write complete sectors without a read-for-ownership request."""
        self.advance(issue)
        visible = rational(visible)
        if visible < issue:
            raise ValueError("Store visibility precedes issue")
        for sector in self.sectors(detail, warp, wave):
            previous = self.pending.get(sector)
            if previous and previous["ready"] > issue:
                raise ValueError("Store must respect same-sector memory order")
            self.counts["store_sectors"] += 1
            self.counts["store_l1_hit_sectors" if sector in self.caches[
                "l1"] else "store_l1_miss_sectors"] += 1
            self.pending[sector] = dict(kind="write", ready=visible)

    def summary(self):
        """Return capacity state and traffic without a fabricated drain cost."""
        return dict(
            counts=dict(sorted(self.counts.items())),
            resident_sectors={key: len(value)
                              for key, value in self.caches.items()},
            pending_sectors=len(self.pending),
            global_time=self.time,
            qualifications=[
                "Fully associative sector LRU is a cache organization "
                "hypothesis, not a measured set/index map.",
                "Per-SM L2 capacity share is nominal, not physical ownership.",
                "Outstanding fills have unlimited capacity in this "
                "resource-ceiling approximation; queue stalls omitted.",
                "Dirty eviction and write-through traffic are counted; "
                "downstream bandwidth, writeback availability and drain "
                "are unmodeled. This is not a coherence timing proof.",
            ],
        )
