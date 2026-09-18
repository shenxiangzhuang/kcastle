--------------------------- MODULE HtmlScroll ---------------------------
EXTENDS Integers, FiniteSets
CONSTANT Fault
ASSUME Fault \in {"none", "phantom", "double", "short", "sidebar", "reverse", "replay", "outside"}

\* The dominant wheel axis is abstracted to one signed dimension.
\* 1 = nested widget, 2 = document viewport, 3 = transcript.
Layers == 1..3
Inner == 1..2
VARIABLES limit, pos, before, phantom, mode, inside, direction, amount,
          phase, candidate, owner, deliveries
vars == <<limit, pos, before, phantom, mode, inside, direction, amount,
          phase, candidate, owner, deliveries>>
environment == <<limit, phantom, mode, inside, direction, amount>>

CanMove(p, i) == IF direction = 1 THEN p[i] < limit[i] ELSE p[i] > 0
Advance(p, i) ==
    LET next == p[i] + direction * amount
    IN [p EXCEPT ![i] = IF next < 0 THEN 0 ELSE IF next > limit[i] THEN limit[i] ELSE next]
Eligible == {i \in Layers : CanMove(before, i) /\ (IF inside THEN (i \in Inner \/ mode = "inline") ELSE i = 3)}
ExpectedOwner == IF Eligible = {} THEN 0 ELSE CHOOSE i \in Eligible : \A j \in Eligible : i <= j
ExpectedPosition == IF ExpectedOwner = 0 THEN before ELSE Advance(before, ExpectedOwner)
Moved == {i \in Layers : pos[i] # before[i]}

Init ==
    /\ limit \in [Layers -> 0..2]
    /\ pos \in [Layers -> 0..2]
    /\ \A i \in Layers : pos[i] <= limit[i]
    /\ before = pos
    \* False-positive geometry/CSS hints must not count as consumed input.
    /\ phantom \in SUBSET Inner
    /\ mode \in {"inline", "sidebar"}
    /\ inside = FALSE /\ direction = 1 /\ amount = 1
    /\ phase = "idle" /\ candidate = 1 /\ owner = 0 /\ deliveries = 0

Begin(at, dir, delta) ==
    /\ phase \in {"idle", "done"}
    /\ inside' = at /\ direction' = dir /\ amount' = delta
    /\ before' = pos /\ owner' = 0 /\ deliveries' = 0 /\ candidate' = 1
    /\ phase' = IF at THEN "inner" ELSE IF Fault = "outside" THEN "done" ELSE "pending"
    /\ UNCHANGED <<limit, pos, phantom, mode>>

TryInner ==
    /\ phase = "inner"
    /\ LET moves == CanMove(pos, candidate)
           claimed == moves \/ candidate \in phantom
           consumed == moves \/ (Fault = "phantom" /\ claimed)
       IN /\ pos' = IF moves THEN Advance(pos, candidate) ELSE pos
          /\ owner' = IF consumed THEN candidate ELSE owner
          /\ phase' = IF consumed THEN (IF Fault = "double" THEN "pending" ELSE "done")
                       ELSE IF candidate = 1 THEN "inner"
                       ELSE IF Fault = "short" \/ (mode = "sidebar" /\ Fault # "sidebar")
                            THEN "done" ELSE "pending"
          /\ candidate' = IF ~consumed /\ candidate = 1 THEN 2 ELSE candidate
    /\ UNCHANGED <<environment, before, deliveries>>

Deliver ==
    /\ phase = "pending"
    /\ pos' = IF Fault = "reverse"
              THEN [pos EXCEPT ![3] = IF direction = 1 THEN 0 ELSE limit[3]]
              ELSE Advance(pos, 3)
    /\ owner' = IF CanMove(pos, 3) THEN 3 ELSE owner
    /\ deliveries' = deliveries + 1
    /\ phase' = IF Fault = "replay" /\ deliveries = 0 THEN "pending" ELSE "done"
    /\ UNCHANGED <<environment, before, candidate>>

Next == (\E at \in BOOLEAN, dir \in {-1, 1}, delta \in 1..2 : Begin(at, dir, delta))
        \/ TryInner \/ Deliver
UnfairSpec == Init /\ [][Next]_vars
Spec == UnfairSpec /\ WF_vars(TryInner) /\ WF_vars(Deliver)

TypeOK ==
    /\ limit \in [Layers -> 0..2] /\ pos \in [Layers -> 0..2] /\ before \in [Layers -> 0..2]
    /\ phantom \subseteq Inner /\ mode \in {"inline", "sidebar"} /\ inside \in BOOLEAN
    /\ direction \in {-1, 1} /\ amount \in 1..2
    /\ phase \in {"idle", "inner", "pending", "done"}
    /\ candidate \in Inner /\ owner \in 0..3 /\ deliveries \in 0..2
WithinBounds == \A i \in Layers : pos[i] >= 0 /\ pos[i] <= limit[i]
NoDoubleConsumption == Cardinality(Moved) <= 1 /\ deliveries <= 1
SidebarIsolated == (inside /\ mode = "sidebar") => pos[3] = before[3] /\ deliveries = 0
OutsidePreservesInner == ~inside => \A i \in Inner : pos[i] = before[i]
InnerFirst == (phase = "done" /\ ExpectedOwner \in Inner) => owner = ExpectedOwner
NoSwallowedWheel == (phase = "done" /\ Eligible # {}) => Moved # {}
CorrectDirection == \A i \in Moved : (pos[i] - before[i]) * direction > 0
MatchesContract == phase = "done" => pos = ExpectedPosition /\ owner = ExpectedOwner
EventSettles == (phase \in {"inner", "pending"}) ~> (phase = "done")

\* Intentionally false invariants: self-test requires a trace reaching each case.
ShortDownReachable == ~(phase = "done" /\ inside /\ limit[1] = 0 /\ limit[2] = 0 /\ direction = 1 /\ owner = 3)
ShortUpReachable == ~(phase = "done" /\ inside /\ limit[1] = 0 /\ limit[2] = 0 /\ direction = -1 /\ owner = 3)
InnerReachable == ~(phase = "done" /\ owner = 2 /\ pos[3] = before[3])
SidebarBoundaryReachable == ~(phase = "done" /\ inside /\ mode = "sidebar" /\ owner = 0 /\ CanMove(before, 3))
=============================================================================
