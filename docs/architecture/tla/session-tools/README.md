# Session/tool protocol model

The executable [TLA+ model](SessionTools.tla) checks a small
slice of [Session v2](../../session.md): tool authorization, durable
dispatch intent, external execution, commit receipts, ordered result attachment,
and cancellation/crash recovery. It is a design check alongside the Rust tests;
it is not a proof that the implementation refines the model.

Run `just tla-check session-tools` and `just tla-self-test session-tools` from the
repository root. Shared prerequisites, output locations, and model conventions are
in the [TLA+ guide](../README.md). The checks below are implemented in [self-test](self-test).

## Model boundary

The checked configuration has one session, one owner, one run/turn/step, two
non-idempotent tools in declaration order, and at most two crashes. The initial
state is immediately after assistant completion and tool declaration. Successful
completion of this tool batch ends the modeled run; additional model turns are
outside the slice. Authorization is abstracted to allowed/denied/pending/aborted;
automatic authorization and unavailable tools do not have separate states.

`durable` is a snapshot of committed facts; `live` is the owner's applied snapshot.
`Commit` changes only durable facts and the durable revision. Receiving or resolving
the receipt advances `live`; a crash can intervene between those actions. Restart
reloads durable facts before recovery. `external` describes what tools actually do,
independently of what the journal knows. External completion remains possible after
owner death or cancellation, but a dead runner cannot send observations to a new owner.
`known` and `executions` are history variables used to check outcome preservation and
dispatch counts; the application does not need to store them.

Assumptions and exclusions are intentional:

- SQLite transactions are atomic and there is one writer. This model does not verify
  SQLite, WAL, OS locks, transaction digests, `tx_id` reuse, or store idempotency.
  Resolving the one outstanding commit is modeled as retrieving the correct receipt.
- A pre-commit failure leaves durable facts unchanged and enters the same stopping
  path as cancellation. Persistent storage unavailability is outside the model.
- Recovery classifies dispatched work without a durable finish as `unknown`, even
  if external work has actually succeeded or has not started. A durable success or
  error survives the finish/result-attachment gap. Recovery does not replay tools.
- New attempts, stale correlation IDs across runs, input admission/steering,
  model requests, compaction, output payloads, UI publication, and projection replay
  are not covered. In particular, `TerminalClosed` covers tools and run/turn/step,
  not the excluded request and compaction lifecycles.
- Tool execution count means harness dispatches for these fixed call identities.
  It says nothing about retries inside a tool/provider or side effects within one call.

There is no state constraint cutting off paths. The actions have finite progress
and the crash budget is finite, so the reachable graph terminates. This still checks
only this finite abstraction, not arbitrary tool counts or unbounded crash histories.

## Properties and implementation mapping

| Property | Meaning | Implementation |
| --- | --- | --- |
| `TypeOK` | All reachable state has the declared shape | Model sanity check |
| `ReceiptConsistency` | Live state never leads storage, and a ready owner has applied the current revision | `AgentLoop::commit_planned`, `acquire_writer_and_reload` in [agent_loop.rs](../../../../crates/agent/src/agent_loop.rs) |
| `AuthorizedEffects` | Each actual dispatch has durable authorization and dispatch intent | `AgentLoop::execute_tools` |
| `NoRepeatedExecution` | Each modeled call is dispatched at most once, including recovery | `execute_tools`, `SessionMachine::plan_recovery` in [machine.rs](../../../../crates/agent/src/session/machine.rs) |
| `PreserveDurableOutcomes` | Recovery cannot downgrade a committed success/error | `SessionMachine::plan_recovery` |
| `OrderedAttachment` | Attached results form a prefix of declaration order | `AgentLoop::attach_ready_results`, `plan_recovery` |
| `ResultConsistency` | Attached statuses agree with durable finishes/authorization | `SessionMachine` event and result validation |
| `TerminalClosed` | The terminal commit closes the modeled lifecycles and resolves all tools | `plan_recovery`, `AgentLoop::terminate_after_error` |
| `StopSettles` | A stop request or crash eventually reaches a durable terminal state | Conditional progress requirement for cleanup/reopen |

`StopSettles` uses weak fairness for receipt delivery/resolution, restart, and terminal
commit: if an action stays enabled, it eventually executes. Combined with the crash
budget, this assumes the owner is eventually restarted and storage/cleanup eventually
make progress. It does not claim that the desktop automatically reopens a session or
retries a failed cleanup. No fairness is assumed for tool completion or user approval,
so a hanging tool does not prevent the modeled cancellation path from settling.

Existing implementation checks include:

- `ambiguous_commit_is_resolved_and_applied_exactly_once` and
  `machine_does_not_advance_when_commit_fails_before_sqlite_commit` in `agent_loop.rs`.
- `tool_finishes_follow_observation_order_but_results_attach_in_call_order` in `agent_loop.rs`.
- `recovery_preserves_a_durable_tool_outcome_across_the_finish_result_gap`,
  `recovery_distinguishes_finished_and_in_flight_parallel_tools`, and
  `recovery_is_one_batch_and_idempotent_after_apply` in `machine.rs`.
- `committed_transaction_is_resolved_after_receipt_is_lost` and
  `hard_kill_rolls_back_and_releases_writer` in [store.rs](../../../../crates/agent/src/session/store.rs),
  which exercise storage behavior assumed by the model.

## Sensitivity checks

`tla-self-test` checks four deliberately broken protocols. The `Fault` constant is
`"none"` for the real model; mutations exist only to test its sensitivity.

| Mutation | Expected violation |
| --- | --- |
| Dispatch after authorization but before durable intent | `AuthorizedEffects` |
| Automatically retry an uncertain, already dispatched tool | `NoRepeatedExecution` |
| Recover every dispatched tool as unknown, ignoring durable finishes | `PreserveDurableOutcomes` |
| Commit terminal state while leaving the step open | `TerminalClosed` |

For example, the retry mutation produces this counterexample: authorize, commit intent,
dispatch, cancel, commit an unknown result and close the run, receive the receipt,
then retry. The execution count becomes two. These are faults intentionally added to
the model, not bugs discovered in the current implementation.

Four additional probes require reachable examples of a lost receipt, tool 2 finishing
durably before tool 1 with no results yet attached, unknown status coexisting with an
actual external completion, and normal completion without cancellation. They use
deliberately false invariants; finding a counterexample is success. A final check
removes fairness and requires a `StopSettles` liveness counterexample. The script
checks both TLC's exit code and the named violation, so parser/runtime errors cannot
be mistaken for successful sensitivity checks.

Architecture/model maintenance guidance lives in [AGENTS.md](../../../../AGENTS.md).
