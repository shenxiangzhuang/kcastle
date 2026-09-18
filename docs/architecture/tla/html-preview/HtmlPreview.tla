--------------------------- MODULE HtmlPreview ---------------------------
EXTENDS Naturals, FiniteSets
CONSTANT Fault
Pages == {"a", "b"}
Sessions == {"A", "B"}
Sites == {"inline", "sidebar"}
VARIABLES session, version, live, visible, covered, displayed, pending, measured, expanded
vars == <<session, version, live, visible, covered, displayed, pending, measured, expanded>>
Token(p, site) == [session |-> session, page |-> p, version |-> version[p], site |-> site]
Mounted(next, large) == {Token(p, "inline") : p \in next}
                       \cup IF large = "none" THEN {} ELSE {Token(large, "sidebar")}
Init == /\ session = "A" /\ version = [p \in Pages |-> 0]
        /\ live = {} /\ visible = {} /\ covered = FALSE
        /\ displayed = {} /\ pending = {} /\ measured = {} /\ expanded = "none"
Frame(next, cover, large) ==
    /\ expanded' = large
    /\ visible' = next /\ covered' = cover
    /\ live' = {t \in live : t.site = "inline"} \cup Mounted(next, large)
    /\ displayed' = IF cover THEN {}
                      ELSE IF Fault = "exclusive" /\ large # "none"
                           THEN Mounted(next \ {large}, large)
                      ELSE Mounted(next, large)
    /\ measured' = {t \in measured : t.site = "inline" \/ t.page = large}
    /\ UNCHANGED <<session, version, pending>>
Queue(t) ==
    /\ t \in live /\ Cardinality(pending) < 2 /\ pending' = pending \cup {t}
    /\ UNCHANGED <<session, version, live, visible, covered, displayed, measured, expanded>>
Receive(t) ==
    /\ t \in pending /\ pending' = pending \ {t}
    /\ measured' = IF Fault = "stale" \/ (t = Token(t.page, t.site) /\ t \in live)
                    THEN {m \in measured : m.page # t.page \/ m.site # t.site} \cup {t} ELSE measured
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
        \/ (\E p \in Pages : Rewrite(p))
        \/ (\E t \in live : Queue(t)) \/ (\E t \in pending : Receive(t)) \/ Switch
Spec == Init /\ [][Next]_vars
CurrentOnly == \A t \in live \cup displayed \cup measured : t = Token(t.page, t.site)
ClippedVisibility == displayed \subseteq Mounted(visible, expanded) /\ (covered => displayed = {})
MountedDocumentsVisible == ~covered => live \cap Mounted(visible, expanded) \subseteq displayed
HiddenStateRetained == [][(UNCHANGED <<session, version>>) => {t \in live : t.site = "inline"} \subseteq live']_vars
MultiplePreviewsReachable == Cardinality(displayed) < 2
SidebarAndInlineReachable == ~(\E p \in Pages : Token(p, "inline") \in displayed /\ Token(p, "sidebar") \in displayed)
=============================================================================
