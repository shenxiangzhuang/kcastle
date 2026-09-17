--------------------------- MODULE HtmlPreview ---------------------------
EXTENDS Naturals, FiniteSets
CONSTANT Fault
Pages == {"a", "b"}
Sessions == {"A", "B"}
VARIABLES session, version, live, visible, covered, displayed, pending, measured, expanded
vars == <<session, version, live, visible, covered, displayed, pending, measured, expanded>>
Token(p) == [session |-> session, page |-> p, version |-> version[p]]
Init == /\ session = "A" /\ version = [p \in Pages |-> 0]
        /\ live = {} /\ visible = {} /\ covered = FALSE
        /\ displayed = {} /\ pending = {} /\ measured = {} /\ expanded = "none"
Frame(next, cover, large) ==
    /\ expanded' = large
    /\ visible' = next /\ covered' = cover
    /\ live' = live \cup {Token(p) : p \in next} \cup IF large = "none" THEN {} ELSE {Token(large)}
    /\ displayed' = IF cover THEN {} ELSE IF large = "none" THEN {Token(p) : p \in next} ELSE {Token(large)}
    /\ UNCHANGED <<session, version, pending, measured>>
Queue(p) ==
    /\ Token(p) \in live /\ pending' = pending \cup {Token(p)}
    /\ UNCHANGED <<session, version, live, visible, covered, displayed, measured, expanded>>
Receive(t) ==
    /\ t \in pending /\ pending' = pending \ {t}
    /\ measured' = IF Fault = "stale" \/ t = Token(t.page)
                    THEN {m \in measured : m.page # t.page} \cup {t} ELSE measured
    /\ UNCHANGED <<session, version, live, visible, covered, displayed, expanded>>
Rewrite(p) ==
    /\ version[p] = 0 /\ version' = [version EXCEPT ![p] = 1]
    /\ live' = {t \in live : t.page # p}
    /\ displayed' = {t \in displayed : t.page # p}
    /\ measured' = {t \in measured : t.page # p}
    /\ UNCHANGED <<session, visible, covered, pending, expanded>>
Switch ==
    /\ session = "A" /\ session' = "B" /\ expanded' = "none"
    /\ live' = {} /\ visible' = {} /\ displayed' = {} /\ measured' = {}
    /\ UNCHANGED <<version, covered, pending>>
Next == (\E next \in SUBSET Pages : \E cover \in BOOLEAN : \E large \in Pages \cup {"none"} : Frame(next, cover, large))
        \/ (\E p \in Pages : Queue(p) \/ Rewrite(p))
        \/ (\E t \in pending : Receive(t)) \/ Switch
Spec == Init /\ [][Next]_vars
CurrentOnly == \A t \in live \cup displayed \cup measured : t = Token(t.page)
ClippedVisibility == \A t \in displayed : ~covered /\ IF expanded = "none" THEN t.page \in visible ELSE t.page = expanded
HiddenStateRetained == [][(UNCHANGED <<session, version>>) => live \subseteq live']_vars
MultiplePreviewsReachable == Cardinality(displayed) < 2
=============================================================================
