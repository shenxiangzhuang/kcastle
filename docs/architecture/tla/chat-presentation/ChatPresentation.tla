------------------------ MODULE ChatPresentation ------------------------
EXTENDS Naturals, FiniteSets
CONSTANT Fault
Blocks == {"a", "b"}
Sessions == {"A", "B"}
Budget == 2
VARIABLES session, generation, wanted, visible, ready, cache, busy, work, cancelled, liveJobs
vars == <<session, generation, wanted, visible, ready, cache, busy, work, cancelled, liveJobs>>
Init == /\ session = "A" /\ generation = 0 /\ wanted = {} /\ visible = {} /\ ready = {} /\ cache = {}
        /\ busy = FALSE /\ work = [session |-> "A", block |-> "a", generation |-> 0]
        /\ cancelled = FALSE /\ liveJobs = 0
Current(values, demand) == {r \in values : r.session = session /\ r.generation = generation /\ r.block \in demand}
Demand(next, onscreen) ==
    /\ wanted' = next /\ visible' = onscreen
    /\ ready' = Current(cache, onscreen)
    /\ cancelled' = (cancelled \/ (busy /\ work.block \notin next))
    /\ UNCHANGED <<session, generation, cache, busy, work, liveJobs>>
Switch(next) ==
    /\ next # session /\ session' = next
    /\ wanted' = {} /\ visible' = {} /\ ready' = {} /\ cancelled' = busy
    /\ UNCHANGED <<generation, cache, busy, work, liveJobs>>
Invalidate ==
    /\ generation < 2 /\ generation' = generation + 1
    /\ wanted' = {} /\ visible' = {} /\ ready' = {} /\ cancelled' = busy
    /\ UNCHANGED <<session, cache, busy, work, liveJobs>>
Start(block) ==
    /\ (~busy \/ Fault = "parallel") /\ liveJobs < 2
    /\ block \in wanted /\ Current(cache, {block}) = {}
    /\ busy' = TRUE /\ cancelled' = FALSE
    /\ work' = [session |-> session, block |-> block, generation |-> generation]
    /\ liveJobs' = liveJobs + 1
    /\ UNCHANGED <<session, generation, wanted, visible, ready, cache>>
Fresh == ~cancelled /\ work.session = session /\ work.generation = generation /\ work.block \in wanted
\* Equal-size units abstract the byte budget. Admission may discard reusable entries,
\* but never the current working set. LRU order and TTL timing are checked in Rust.
Admitted == IF Fault = "unbounded-cache" \/ Cardinality(cache \cup {work}) <= Budget
            THEN cache \cup {work}
            ELSE IF Cardinality(ready \cup {work}) <= Budget THEN ready \cup {work} ELSE cache
Finish ==
    /\ busy
    /\ cache' = IF Fault = "stale" \/ Fresh THEN Admitted ELSE cache
    /\ ready' = IF Fault = "stale" THEN ready \cup {work} ELSE Current(cache', visible)
    /\ busy' = FALSE /\ cancelled' = FALSE /\ liveJobs' = liveJobs - 1
    /\ UNCHANGED <<session, generation, wanted, visible, work>>
\* Semantic index publication rekeys rows; indices themselves are abstracted away.
FinishIndex ==
    /\ busy
    /\ wanted' = IF Fresh THEN {} ELSE wanted
    /\ visible' = IF Fresh THEN {} ELSE visible
    /\ ready' = IF Fresh THEN {} ELSE ready
    /\ busy' = FALSE /\ cancelled' = FALSE /\ liveJobs' = liveJobs - 1
    /\ UNCHANGED <<session, generation, cache, work>>
Evict(remaining) ==
    /\ remaining \subseteq cache
    /\ (Fault = "evict-live" \/ ready \subseteq remaining)
    /\ cache' = remaining
    /\ UNCHANGED <<session, generation, wanted, visible, ready, busy, work, cancelled, liveJobs>>
Next == (\E next \in SUBSET Blocks : \E onscreen \in SUBSET next : Demand(next, onscreen)) \/ Invalidate
        \/ (\E next \in Sessions : Switch(next))
        \/ (\E block \in Blocks : Start(block)) \/ Finish \/ FinishIndex
        \/ (\E remaining \in SUBSET cache : Evict(remaining))
Spec == Init /\ [][Next]_vars /\ WF_vars(Finish)
CurrentOnly == \A r \in ready : r.session = session /\ r.generation = generation /\ r.block \in visible
BoundedWorker == liveJobs <= 1
BoundedReady == Cardinality(ready) <= Cardinality(visible)
BoundedCache == Cardinality(cache) <= Budget
CachedReady == ready \subseteq cache
CancelledSettles == cancelled ~> ~busy
NoReady == ready = {}
=============================================================================
