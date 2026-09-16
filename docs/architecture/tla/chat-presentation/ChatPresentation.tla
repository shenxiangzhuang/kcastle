------------------------ MODULE ChatPresentation ------------------------
EXTENDS Naturals, FiniteSets
CONSTANT Fault
Blocks == {"a", "b"}
VARIABLES generation, wanted, ready, busy, work, cancelled, liveJobs
vars == <<generation, wanted, ready, busy, work, cancelled, liveJobs>>
Init == /\ generation = 0 /\ wanted = {} /\ ready = {}
        /\ busy = FALSE /\ work = [block |-> "a", generation |-> 0]
        /\ cancelled = FALSE /\ liveJobs = 0
Demand(next) ==
    /\ wanted' = next
    /\ ready' = IF Fault = "keep-offscreen" THEN ready
                ELSE {r \in ready : r.block \in next}
    /\ cancelled' = (cancelled \/ (busy /\ work.block \notin next))
    /\ UNCHANGED <<generation, busy, work, liveJobs>>
Invalidate ==
    /\ generation < 2
    /\ generation' = generation + 1
    /\ wanted' = {} /\ ready' = {}
    /\ cancelled' = busy
    /\ UNCHANGED <<busy, work, liveJobs>>
Start(block) ==
    /\ (~busy \/ Fault = "parallel") /\ liveJobs < 2
    /\ block \in wanted
    /\ ~\E r \in ready : r.block = block
    /\ busy' = TRUE /\ cancelled' = FALSE
    /\ work' = [block |-> block, generation |-> generation]
    /\ liveJobs' = liveJobs + 1
    /\ UNCHANGED <<generation, wanted, ready>>
Finish ==
    /\ busy
    /\ ready' = IF Fault = "stale" \/
                   (~cancelled /\ work.generation = generation /\ work.block \in wanted)
                THEN ready \cup {work} ELSE ready
    /\ busy' = FALSE /\ cancelled' = FALSE /\ liveJobs' = liveJobs - 1
    /\ UNCHANGED <<generation, wanted, work>>
\* Semantic index publication rekeys rows and clears demand until the next frame.
FinishIndex ==
    /\ busy
    /\ wanted' = IF ~cancelled /\ work.generation = generation /\ work.block \in wanted
                  THEN {} ELSE wanted
    /\ ready' = IF ~cancelled /\ work.generation = generation /\ work.block \in wanted
                 THEN {} ELSE ready
    /\ busy' = FALSE /\ cancelled' = FALSE /\ liveJobs' = liveJobs - 1
    /\ UNCHANGED <<generation, work>>
Next == (\E next \in SUBSET Blocks : Demand(next)) \/ Invalidate
        \/ (\E block \in Blocks : Start(block)) \/ Finish \/ FinishIndex
Spec == Init /\ [][Next]_vars /\ WF_vars(Finish)
CurrentOnly == \A r \in ready : r.generation = generation /\ r.block \in wanted
BoundedWorker == liveJobs <= 1
BoundedReady == Cardinality(ready) <= Cardinality(wanted)
CancelledSettles == cancelled ~> ~busy
NoReady == ready = {}
=============================================================================
