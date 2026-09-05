"""Schedule qualified instruction hierarchy requests and pending fills."""

from collections import Counter, OrderedDict
from fractions import Fraction


def cycles(value):
    """Read nonnegative exact cycles from a supplied physical hypothesis."""
    if isinstance(value, dict):
        value = Fraction(value["numerator"], value["denominator"])
    elif isinstance(value, (int, str, Fraction)):
        value = Fraction(value)
    else:
        raise ValueError("Instruction services require exact rational cycles")
    if value < 0:
        raise ValueError("Instruction services cannot be negative")
    return value


class NominalInstructionCache:
    """Keep source-PC instruction residency distinct from issue readiness.

    Four supplied domain labels map scheduler partitions to cache instances.
    These labels express hypotheses, not a discovered physical SM mapping.
    Whole-path readiness is charged once; request queuing is separate.
    """

    def __init__(self, specification):
        self.specification = specification
        if specification.get("mode") != "hierarchy":
            raise ValueError("Instruction hierarchy mode is required")
        if not specification.get("provenance") or not specification.get(
                "assumption"):
            raise ValueError("Instruction services need physical provenance")
        if specification.get("outstanding_fills") != (
                "unlimited_capacity_ceiling"):
            raise ValueError("Unidentified fill queue needs an explicit ceiling")
        if specification.get("fetch_policy") != "next_pc_demand":
            raise ValueError("Only next-PC demand requests are represented")
        self.levels = specification["levels"]
        if not self.levels or len({x["name"] for x in self.levels}) != len(
                self.levels):
            raise ValueError("Instruction levels need distinct names")
        self.resident = {}
        self.pending = {}
        self.resource_ready = {}
        self.counts = Counter()
        self.reservations = Counter()
        self.clock = Fraction(0)
        self.serial = 0
        previous_line = 16
        for level in self.levels:
            size, capacity = level["line_bytes"], level["capacity_bytes"]
            if (type(size) is not int or size < previous_line
                    or size % previous_line or type(capacity) is not int
                    or capacity < 0 or capacity % size):
                raise ValueError("Nested instruction lines/capacities misalign")
            previous_line = size
            if (len(level["partition_domains"]) != 4
                    or any(not isinstance(x, str) or not x
                           for x in level["partition_domains"])):
                raise ValueError("Four explicit partition domain labels required")
            if cycles(level["request_interval_cycles"]) <= 0:
                raise ValueError("Finite request intervals must be positive")
            cycles(level["path_ready_cycles"])
            for domain in level["partition_domains"]:
                key = (level["name"], domain)
                self.resident[key] = OrderedDict()
        backing = specification["backing"]
        if len(backing["partition_domains"]) != 4:
            raise ValueError("Backing requires four explicit domain labels")
        if cycles(backing["request_interval_cycles"]) <= 0:
            raise ValueError("Backing initiation must be positive")
        cycles(backing["path_ready_cycles"])
        initial = specification["initial_state"]
        if initial not in ("cold", "explicit_seed_lines"):
            raise ValueError("Instruction initial state must be explicit")
        if initial == "explicit_seed_lines":
            for item in specification["seed_lines"]:
                level = next(x for x in self.levels if x["name"] == item["level"])
                key = (level["name"], item["domain"])
                if key not in self.resident:
                    raise ValueError("Seed line has an unknown cache domain")
                line = item["line"]
                if type(line) is not int or line < 0:
                    raise ValueError("Seed line needs a nonnegative line index")
                self.install(key, line, level)

    def install(self, key, line, level):
        """Install one arrived immutable line under fully associative LRU."""
        cache = self.resident[key]
        capacity = level["capacity_bytes"] // level["line_bytes"]
        if not capacity:
            return
        cache.pop(line, None)
        cache[line] = None
        while len(cache) > capacity:
            cache.popitem(last=False)
            self.counts[level["name"] + ":evictions"] += 1

    def advance(self, time):
        """Apply arrived fills in timestamp and request order."""
        time = cycles(time)
        if time < self.clock:
            raise ValueError("Instruction cache time cannot run backwards")
        self.clock = time
        arrived = sorted(
            ((value["ready"], value["serial"], key, value)
             for key, value in self.pending.items() if value["ready"] <= time),
            key=lambda item: item[:2],
        )
        for _, _, key, value in arrived:
            self.install(key[:2], key[2], value["level"])
            del self.pending[key]

    def earliest(self, partition, eligible):
        """Inspect front-end availability without changing cache state."""
        level = self.levels[0]
        resource = (level["name"], level["partition_domains"][partition])
        return max(cycles(eligible), self.resource_ready.get(resource, 0))

    def reserve(self, name, domain, interval, time):
        """Reserve one request resource; return its queue-delayed start."""
        key = (name, domain)
        start = max(time, self.resource_ready.get(key, 0))
        interval = cycles(interval)
        self.resource_ready[key] = start + interval
        self.reservations[name + ":" + domain] += interval
        return start

    def request(self, partition, pc, time):
        """Commit one selected fetch, merging pending immutable line fills."""
        if type(pc) is not int or pc < 0 or pc % 16:
            raise ValueError("Source instruction PC must be a 16-byte slot")
        time = cycles(time)
        if time != self.earliest(partition, time):
            raise ValueError("Fetch started before its front-end resource")
        self.advance(time)
        queued = time
        missed = []
        path = None
        merged = False
        for level in self.levels:
            name = level["name"]
            domain = level["partition_domains"][partition]
            key = (name, domain)
            line = pc // level["line_bytes"]
            self.counts[name + ":requests"] += 1
            queued = self.reserve(name, domain,
                                  level["request_interval_cycles"], queued)
            cache = self.resident[key]
            pending = self.pending.get((*key, line))
            if line in cache:
                cache.move_to_end(line)
                self.counts[name + ":hits"] += 1
                path = name
                ready = queued + cycles(level["path_ready_cycles"])
                break
            if pending is not None:
                self.counts[name + ":pending_merges"] += 1
                path = name + ":pending"
                ready = max(pending["ready"], queued + cycles(
                    self.levels[0]["path_ready_cycles"]))
                merged = True
                break
            self.counts[name + ":misses"] += 1
            missed.append((level, key, line))
        else:
            backing = self.specification["backing"]
            queued = self.reserve(
                "backing", backing["partition_domains"][partition],
                backing["request_interval_cycles"], queued)
            ready = queued + cycles(backing["path_ready_cycles"])
            path = "backing"
            self.counts["backing:requests"] += 1
            self.counts["backing:bytes"] += self.levels[-1]["line_bytes"]
        for level, key, line in missed:
            self.serial += 1
            self.pending[(*key, line)] = dict(
                ready=ready, serial=self.serial, level=level)
            self.counts[level["name"] + ":fills"] += 1
            self.counts[level["name"] + ":fill_bytes"] += level["line_bytes"]
        self.counts["front:requests"] += 1
        return dict(pc=pc, request=time, ready=ready, path=path,
                    queue_cycles=queued - time, pending_merge=merged)

    def summary(self):
        """Report traffic, resources and explicit cache domain hypotheses."""
        return dict(
            counts=dict(sorted(self.counts.items())),
            resource_reserved_cycles=dict(sorted(self.reservations.items())),
            pending_fills=len(self.pending),
            resident_lines={name + ":" + domain: len(lines)
                            for (name, domain), lines in self.resident.items()},
            specification=self.specification,
            qualifications=[
                "Fully associative immutable lines; no set-index inference",
                "Whole-path readiness includes lookup/return service once",
                "Level request lookups are logically concurrent; only queue "
                "waiting is added to whole-path readiness",
                "Pending fills occupy an explicit unlimited-queue ceiling",
                "Each modeled SM uses the supplied nominal domain budget; "
                "other-SM traffic and physical GPC membership are not inferred",
                "No stream prefetch depth or fitted miss penalty",
            ],
        )
