------------------------ MODULE ChatPresentation ------------------------
EXTENDS Naturals, FiniteSets
CONSTANT Fault
Blocks == {"a", "b"}
Sessions == {"A", "B"}
Budget == 2
VARIABLES session, generation, target, wanted, visible, ready, cache, busy, work, cancelled, liveJobs
vars == <<session, generation, target, wanted, visible, ready, cache, busy, work, cancelled, liveJobs>>
Init == /\ session = "A" /\ generation = 0 /\ target = 0 /\ wanted = {} /\ visible = {} /\ ready = {} /\ cache = {}
        /\ busy = FALSE /\ work = [session |-> "A", block |-> "a", generation |-> 0]
        /\ cancelled = FALSE /\ liveJobs = 0
Current(values, demand) == {r \in values : r.session = session /\ r.generation = generation /\ r.block \in demand}
Demand(next, onscreen) ==
    /\ wanted' = next /\ visible' = onscreen
    /\ ready' = Current(cache, onscreen)
    /\ cancelled' = (cancelled \/ (busy /\ work.block \notin next))
    /\ UNCHANGED <<session, generation, target, cache, busy, work, liveJobs>>
Switch(next) ==
    /\ next # session /\ session' = next
    /\ wanted' = {} /\ visible' = {} /\ ready' = {} /\ cancelled' = busy
    /\ UNCHANGED <<generation, target, cache, busy, work, liveJobs>>
Invalidate ==
    /\ target < 2 /\ generation' = target + 1 /\ target' = target + 1
    /\ wanted' = {} /\ visible' = {} /\ ready' = {} /\ cancelled' = busy
    /\ UNCHANGED <<session, cache, busy, work, liveJobs>>
\* Append changes the desired source, not the currently displayed snapshot.
Append ==
    /\ target < 2 /\ target' = target + 1
    /\ cancelled' = (cancelled \/ (busy /\ work.generation = generation))
    /\ ready' = IF Fault = "flash" THEN {} ELSE ready
    /\ UNCHANGED <<session, generation, wanted, visible, cache, busy, work, liveJobs>>
Start(block) ==
    /\ (~busy \/ Fault = "parallel") /\ liveJobs < 2
    /\ block \in wanted /\ (target # generation \/ Current(cache, {block}) = {})
    /\ busy' = TRUE /\ cancelled' = FALSE
    /\ work' = [session |-> session, block |-> block, generation |-> target]
    /\ liveJobs' = liveJobs + 1
    /\ UNCHANGED <<session, generation, target, wanted, visible, ready, cache>>
Fresh == ~cancelled /\ work.session = session /\ work.block \in wanted
    /\ (work.generation = target \/ (work.generation > generation /\ work.generation < target))
\* Equal-size units abstract the byte budget. Admission may discard reusable entries,
\* but never the current working set. LRU order and TTL timing are checked in Rust.
Admitted == IF Fault = "unbounded-cache" \/ Cardinality(cache \cup {work}) <= Budget
            THEN cache \cup {work}
            ELSE IF Cardinality(ready \cup {work}) <= Budget THEN ready \cup {work} ELSE cache
Finish ==
    /\ busy /\ work.generation = generation
    /\ cache' = IF Fault = "stale" \/ Fresh THEN Admitted ELSE cache
    /\ ready' = IF Fault = "stale" THEN ready \cup {work} ELSE Current(cache', visible)
    /\ busy' = FALSE /\ cancelled' = FALSE /\ liveJobs' = liveJobs - 1
    /\ UNCHANGED <<session, generation, target, wanted, visible, work>>
\* One index job also prepares the demanded replacement. No partial index is shown.
Replacement == {[session |-> session, block |-> b, generation |-> work.generation] : b \in wanted}
FinishUpdate ==
    /\ busy /\ work.generation # generation
    /\ cache' = IF Fresh THEN Replacement ELSE cache
    /\ ready' = IF Fresh THEN {r \in Replacement : r.block \in visible} ELSE ready
    /\ generation' = IF Fresh THEN work.generation ELSE generation
    /\ busy' = FALSE /\ cancelled' = FALSE /\ liveJobs' = liveJobs - 1
    /\ UNCHANGED <<session, target, wanted, visible, work>>
\* Semantic index publication rekeys rows; indices themselves are abstracted away.
FinishIndex ==
    /\ busy /\ work.generation = generation
    /\ wanted' = IF Fresh THEN {} ELSE wanted
    /\ visible' = IF Fresh THEN {} ELSE visible
    /\ ready' = IF Fresh THEN {} ELSE ready
    /\ busy' = FALSE /\ cancelled' = FALSE /\ liveJobs' = liveJobs - 1
    /\ UNCHANGED <<session, generation, target, cache, work>>
Evict(remaining) ==
    /\ remaining \subseteq cache
    /\ (Fault = "evict-live" \/ ready \subseteq remaining)
    /\ cache' = remaining
    /\ UNCHANGED <<session, generation, target, wanted, visible, ready, busy, work, cancelled, liveJobs>>
Next == (\E next \in SUBSET Blocks : \E onscreen \in SUBSET next : Demand(next, onscreen)) \/ Invalidate
        \/ (\E next \in Sessions : Switch(next))
        \/ Append \/ (\E block \in Blocks : Start(block)) \/ Finish \/ FinishIndex \/ FinishUpdate
        \/ (\E remaining \in SUBSET cache : Evict(remaining))
Spec == Init /\ [][Next]_vars /\ WF_vars(Finish \/ FinishUpdate)
CurrentOnly == \A r \in ready : r.session = session /\ r.generation = generation /\ r.block \in visible
DisplayedVersion == generation <= target
NoIntermediateDowngrade == [][(target' > target /\ generation' = generation
    /\ UNCHANGED <<session, wanted, visible>>) => ready' = ready]_vars
BoundedWorker == liveJobs <= 1
BoundedReady == Cardinality(ready) <= Cardinality(visible)
BoundedCache == Cardinality(cache) <= Budget
CachedReady == ready \subseteq cache
CancelledSettles == cancelled ~> ~busy
NoReady == ready = {}
=============================================================================
