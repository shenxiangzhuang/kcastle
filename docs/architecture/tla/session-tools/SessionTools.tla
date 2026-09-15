-------------------------- MODULE SessionTools --------------------------
EXTENDS Naturals, Sequences

CONSTANTS ToolCount, MaxCrashes, Fault
ASSUME /\ ToolCount \in Nat \ {0}
       /\ MaxCrashes \in Nat
       /\ Fault \in {"none", "early-dispatch", "retry-unknown",
                     "forget-finish", "partial-terminal"}

Tools == 1..ToolCount
Order == [i \in Tools |-> i]
Outcomes == {"none", "success", "error", "unknown"}
KnownOutcomes == {"success", "error"}
Lifecycles == {"run", "turn", "step"}
Snapshot == [auth : [Tools -> {"pending", "allowed", "denied", "aborted"}],
             intent : SUBSET Tools, started : SUBSET Tools,
             finish : [Tools -> Outcomes],
             result : [Tools -> Outcomes \cup {"denied", "aborted"}],
             attached : Seq(Tools), open : SUBSET Lifecycles,
             terminal : BOOLEAN]

VARIABLES durable, live, revision, liveRevision, phase,
          external, runners, executions, known, stopping, crashes
storeVars == <<durable, live, revision, liveRevision, phase>>
effectVars == <<external, runners, executions>>
controlVars == <<stopping, crashes>>
vars == <<storeVars, effectVars, known, controlVars>>

Init ==
    /\ durable = [auth |-> [t \in Tools |-> "pending"],
                   intent |-> {}, started |-> {},
                   finish |-> [t \in Tools |-> "none"],
                   result |-> [t \in Tools |-> "none"],
                   attached |-> <<>>, open |-> Lifecycles, terminal |-> FALSE]
    /\ live = durable
    /\ revision = 0 /\ liveRevision = 0 /\ phase = "ready"
    /\ external = [t \in Tools |-> "absent"]
    /\ runners = {} /\ executions = [t \in Tools |-> 0]
    /\ known = [t \in Tools |-> "none"]
    /\ stopping = FALSE /\ crashes = 0

Ready == phase = "ready" /\ ~durable.terminal
Normal == Ready /\ ~stopping

\* SQLite atomicity and one writer are assumptions. Receipt delivery is NOT
\* atomic with commit. No other append may start while its outcome is pending.
Commit(next) ==
    /\ durable' = next /\ revision' = revision + 1
    /\ phase' = "receipt"
    /\ UNCHANGED <<live, liveRevision>>

Authorize(t, decision) ==
    /\ Normal /\ live.auth[t] = "pending"
    /\ Commit([live EXCEPT !.auth[t] = decision])
    /\ UNCHANGED <<effectVars, known, controlVars>>

DispatchIntent ==
    /\ Normal /\ \A t \in Tools : live.auth[t] # "pending"
    /\ live.intent = {}
    /\ LET allowed == {t \in Tools : live.auth[t] = "allowed"}
       IN /\ allowed # {}
          /\ Commit([live EXCEPT !.intent = allowed])
    /\ UNCHANGED <<effectVars, known, controlVars>>

StartTool(t) ==
    /\ Normal /\ live.auth[t] = "allowed" /\ t \notin runners
    /\ (t \in live.intent \/ Fault = "early-dispatch")
    /\ external' = [external EXCEPT ![t] = "running"]
    /\ runners' = runners \cup {t}
    /\ executions' = [executions EXCEPT ![t] = @ + 1]
    /\ UNCHANGED <<storeVars, known, controlVars>>

\* External work can finish during an append, after cancellation, or even
\* after the owner dies. A lost runner cannot deliver to a replacement owner.
FinishExternal(t, outcome) ==
    /\ external[t] = "running"
    /\ external' = [external EXCEPT ![t] = outcome]
    /\ UNCHANGED <<storeVars, runners, executions, known, controlVars>>

RecordStart(t) ==
    /\ Normal /\ t \in runners /\ t \notin live.started
    /\ Commit([live EXCEPT !.started = @ \cup {t}])
    /\ UNCHANGED <<effectVars, known, controlVars>>

RecordFinish(t) ==
    /\ Normal /\ t \in runners /\ t \in live.started
    /\ external[t] \in KnownOutcomes /\ live.finish[t] = "none"
    /\ Commit([live EXCEPT !.finish[t] = external[t]])
    \* History variable: what was durably known BEFORE any later recovery.
    /\ known' = [known EXCEPT ![t] = external[t]]
    /\ UNCHANGED <<effectVars, controlVars>>

AttachNext ==
    /\ Normal /\ Len(live.attached) < ToolCount
    /\ LET t == Len(live.attached) + 1
           result == IF live.auth[t] = "denied" THEN "denied"
                     ELSE live.finish[t]
       IN /\ result \in KnownOutcomes \cup {"denied"}
          /\ Commit([live EXCEPT !.result[t] = result,
                                !.attached = Append(@, t)])
    /\ UNCHANGED <<effectVars, known, controlVars>>

ReceiveReceipt ==
    /\ phase = "receipt"
    /\ live' = durable /\ liveRevision' = revision /\ phase' = "ready"
    /\ UNCHANGED <<durable, revision, effectVars, known, controlVars>>

LoseReceipt ==
    /\ phase = "receipt" /\ phase' = "unknown"
    /\ UNCHANGED <<durable, live, revision, liveRevision,
                   effectVars, known, controlVars>>

\* Lookup of the same outstanding tx_id returns its committed receipt.
\* This abstracts store.resolve; transaction-ID/digest validation is not modeled.
ResolveReceipt ==
    /\ phase = "unknown"
    /\ live' = durable /\ liveRevision' = revision /\ phase' = "ready"
    /\ UNCHANGED <<durable, revision, effectVars, known, controlVars>>

\* Also represents a pre-commit storage failure: no durable transition occurred.
Stop ==
    /\ ~durable.terminal /\ ~stopping /\ phase # "crashed"
    /\ stopping' = TRUE
    /\ UNCHANGED <<storeVars, effectVars, known, crashes>>

Crash ==
    /\ phase # "crashed" /\ crashes < MaxCrashes
    /\ phase' = "crashed" /\ crashes' = crashes + 1
    /\ stopping' = TRUE /\ runners' = {}
    /\ UNCHANGED <<durable, live, revision, liveRevision,
                   external, executions, known>>

Restart ==
    /\ phase = "crashed"
    /\ live' = durable /\ liveRevision' = revision /\ phase' = "ready"
    /\ UNCHANGED <<durable, revision, effectVars, known, controlVars>>

RecoveredFinish(t) ==
    IF t \in live.intent /\
       (live.finish[t] = "none" \/ Fault = "forget-finish")
    THEN "unknown" ELSE live.finish[t]
RecoveredResult(t) ==
    IF live.result[t] # "none" THEN live.result[t]
    ELSE IF live.auth[t] = "denied" THEN "denied"
    ELSE IF t \notin live.intent THEN "aborted"
    ELSE RecoveredFinish(t)

Terminate ==
    /\ Ready /\ (stopping \/ Len(live.attached) = ToolCount)
    /\ Commit([live EXCEPT
         !.auth = [t \in Tools |-> IF live.auth[t] = "pending"
                                  THEN "aborted" ELSE live.auth[t]],
         !.finish = [t \in Tools |-> RecoveredFinish(t)],
         !.result = [t \in Tools |-> RecoveredResult(t)],
         !.attached = Order,
         !.open = IF Fault = "partial-terminal" THEN {"step"} ELSE {},
         !.terminal = TRUE])
    /\ UNCHANGED <<effectVars, known, controlVars>>

\* Negative control: a recovery policy that retries uncertain external work.
RetryUnknown(t) ==
    /\ Fault = "retry-unknown" /\ phase = "ready" /\ durable.terminal
    /\ durable.result[t] = "unknown" /\ executions[t] = 1
    /\ executions' = [executions EXCEPT ![t] = @ + 1]
    /\ external' = [external EXCEPT ![t] = "running"]
    /\ UNCHANGED <<storeVars, runners, known, controlVars>>

Done == phase = "ready" /\ durable.terminal /\ UNCHANGED vars
Next ==
    \/ \E t \in Tools, d \in {"allowed", "denied"} : Authorize(t, d)
    \/ DispatchIntent
    \/ \E t \in Tools : StartTool(t) \/ RecordStart(t) \/ RecordFinish(t)
                         \/ RetryUnknown(t)
    \/ \E t \in Tools, o \in KnownOutcomes : FinishExternal(t, o)
    \/ AttachNext \/ ReceiveReceipt \/ LoseReceipt \/ ResolveReceipt
    \/ Stop \/ Crash \/ Restart \/ Terminate \/ Done

Spec == Init /\ [][Next]_vars
\* Bounded crashes plus weak fairness mean the owner and store eventually
\* make progress. No assumption that a tool finishes or a user approves it.
FairSpec == Spec /\ WF_vars(ReceiveReceipt) /\ WF_vars(ResolveReceipt)
                 /\ WF_vars(Restart) /\ WF_vars(Terminate)

TypeOK ==
    /\ durable \in Snapshot /\ live \in Snapshot
    /\ revision \in Nat /\ liveRevision \in Nat
    /\ phase \in {"ready", "receipt", "unknown", "crashed"}
    /\ external \in [Tools -> {"absent", "running", "success", "error"}]
    /\ runners \subseteq Tools /\ executions \in [Tools -> 0..2]
    /\ known \in [Tools -> KnownOutcomes \cup {"none"}]
    /\ stopping \in BOOLEAN /\ crashes \in 0..MaxCrashes
ReceiptConsistency ==
    /\ liveRevision <= revision /\ revision <= liveRevision + 1
    /\ (liveRevision = revision => live = durable)
    /\ (phase = "ready" => liveRevision = revision)
AuthorizedEffects ==
    \A t \in Tools : executions[t] > 0 =>
        t \in durable.intent /\ durable.auth[t] = "allowed"
NoRepeatedExecution == \A t \in Tools : executions[t] <= 1
PreserveDurableOutcomes ==
    \A t \in Tools : known[t] # "none" => durable.finish[t] = known[t]
OrderedAttachment ==
    /\ Len(durable.attached) <= ToolCount
    /\ durable.attached = SubSeq(Order, 1, Len(durable.attached))
    /\ \A t \in Tools :
        (durable.result[t] # "none") <=> (t <= Len(durable.attached))
ResultConsistency ==
    \A t \in Tools :
        /\ (durable.finish[t] \in KnownOutcomes =>
             t \in durable.started /\ durable.finish[t] = external[t])
        /\ (durable.result[t] \in KnownOutcomes \cup {"unknown"} =>
             durable.result[t] = durable.finish[t])
        /\ (durable.result[t] = "aborted" => t \notin durable.intent)
        /\ (durable.result[t] = "denied" => durable.auth[t] = "denied")
TerminalClosed ==
    durable.terminal =>
        /\ durable.open = {} /\ durable.attached = Order
        /\ \A t \in Tools :
            /\ durable.auth[t] # "pending"
            /\ (t \in durable.intent => durable.finish[t] # "none")
StopSettles == stopping ~> durable.terminal

\* Reachability probes, deliberately false invariants checked separately.
\* They ensure the finite model actually explores the advertised boundaries.
NoLostReceipt == phase # "unknown"
NoParallelFinishGap ==
    ~(ToolCount = 2 /\ durable.finish[2] \in KnownOutcomes
      /\ durable.finish[1] = "none" /\ durable.attached = <<>>)
NoUncertainCompletion ==
    ~(durable.terminal /\ \E t \in Tools :
        durable.result[t] = "unknown" /\ external[t] \in KnownOutcomes)
NoNormalCompletion == ~(durable.terminal /\ ~stopping)
=============================================================================
