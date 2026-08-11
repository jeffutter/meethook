export const meta = {
  name: "ralph-backlog-loop",
  description:
    "Priority-ordered autonomous backlog loop: execute > plan > choose",
  phases: [
    {
      title: "Setup",
      detail: "ensure required backlog statuses exist",
      model: "haiku",
    },
    {
      title: "State",
      detail:
        "detect In Progress / Dev Ready / Needs Plan tickets via backlog CLI",
      model: "haiku",
    },
    { title: "Execute", detail: "run /backlog-execute on one ticket" },
    { title: "Plan", detail: "run /backlog-planner on one ticket" },
    {
      title: "Choose",
      detail:
        "pick next unblocked To Do ticket; route to Dev Ready if already planned, else Needs Plan",
      model: "haiku",
    },
  ],
};

const MAX_ITERATIONS = (() => {
  if (typeof args === "number") return args;
  if (
    args &&
    typeof args === "object" &&
    typeof args.maxIterations === "number"
  )
    return args.maxIterations;
  return 25;
})();

const SETUP_SCHEMA = {
  type: "object",
  properties: {
    statuses: { type: "array", items: { type: "string" } },
    changed: { type: "boolean" },
  },
  required: ["statuses", "changed"],
};

const STATE_SCHEMA = {
  type: "object",
  properties: {
    inProgress: {
      type: "array",
      items: { type: "string" },
      description: "Ticket IDs with status In Progress, in list order",
    },
    devReady: {
      type: "array",
      items: { type: "string" },
      description: "Ticket IDs with status Dev Ready, in list order",
    },
    needsPlan: {
      type: "array",
      items: { type: "string" },
      description: "Ticket IDs with status Needs Plan, in list order",
    },
  },
  required: ["inProgress", "devReady", "needsPlan"],
};

const CHOOSE_SCHEMA = {
  type: "object",
  properties: {
    ticketId: {
      type: ["string", "null"],
      description:
        "The ticket ID that was advanced, or null if no eligible To Do ticket exists",
    },
    destination: {
      type: ["string", "null"],
      description:
        'The status it was moved to: "Dev Ready" if it already carried an implementation plan, "Needs Plan" if it did not, or null if nothing was chosen',
    },
    reason: {
      type: "string",
      description: "Short explanation of the choice or why nothing was chosen",
    },
  },
  required: ["ticketId", "destination", "reason"],
};

const ACTION_SCHEMA = {
  type: "object",
  properties: {
    ticketId: { type: "string" },
    outcome: {
      type: "string",
      description: "One of: completed, blocked-reverted, planned, error",
    },
    summary: {
      type: "string",
      description: "One sentence describing what happened",
    },
  },
  required: ["ticketId", "outcome", "summary"],
};

const SETUP_PROMPT = `You're working in the meethook repo at its root.

This project's backlog work loop needs this pipeline, plus one parking status off to the side:
  To Do -> Needs Plan -> Dev Ready -> In Progress -> Done
  Blocked (where a ticket waits when its dependencies are not yet Done)

Run: backlog config get statuses

Both "Dev Ready" and "Blocked" must be present. If both already are, change nothing.

For any that are missing, edit backlog/config.yml and insert them into the \`statuses\` list:
- "Dev Ready" goes between "Needs Plan" and "In Progress".
- "Blocked" goes immediately after "To Do", so the list reads as work that is on the board but cannot move.

(\`backlog config set\` does not accept the \`statuses\` key, so a direct file edit is the only way to change it — this is the one backlog file you may edit by hand. Never hand-edit task, draft, document, decision, or milestone markdown.) Do not remove, rename, or reorder any other existing statuses.

Report via structured output:
- statuses: the final statuses list (after your edit, if any)
- changed: true if you modified the file, false if it already had everything needed`;

const STATE_PROMPT = `In the meethook repo (repo root, no git submodules), run exactly one command:

  backlog task list --exclude-status "Backlog,To Do,Blocked,Done" --json

That returns versioned JSON of shape {schemaVersion, kind: "task-list", tasks: [{id, title, status, labels, ordinal, ...}]}. Do not add --plain; --json and --plain cannot be combined.

Group the returned tasks by their \`status\` field and report the ticket IDs for each of "In Progress", "Dev Ready", and "Needs Plan", preserving the order they appear in the \`tasks\` array. Any status outside those three should be ignored.`;

function choosePrompt(parked) {
  const parkedClause = parked.length
    ? `\n\nAlso skip these ticket IDs entirely — this run already tried them and they reverted as blocked more than once: ${parked.join(", ")}.`
    : "";
  return `You're working in the meethook repo at its root.

**Pass 1 — fresh work.** Run:

  backlog task list -s "To Do" --ready --sort ordinal --json

\`--ready\` already restricts the result to unblocked tasks whose dependencies are all Done, and \`--sort ordinal\` returns them in the backlog's intended work order, so you do NOT need to inspect dependencies yourself. Do not add --plain; --json and --plain cannot be combined.

The result is {schemaVersion, kind: "task-list", tasks: [{id, title, status, labels, ordinal, ...}]}. Walk \`tasks\` in order and take the first whose \`labels\` array does not contain "no-ralph".${parkedClause}

**Pass 2 — revive blocked work, only if pass 1 found nothing.** Run:

  backlog task list -s "Blocked" --ready --sort ordinal --json

A ticket in "Blocked" was parked by a previous execution attempt because something else had to land first, and its blockers were recorded as dependencies at that time. \`--ready\` is independent of a task's own status, so anything this query returns is a blocked ticket whose blockers are now all Done — it is ready to move again. Apply the same "no-ralph" and parked-ID skips.

Taking blocked work only after fresh work is exhausted keeps parked tickets from crowding out the queue, while still guaranteeing they resume once they can.

If neither pass yields a candidate, change nothing and report ticketId: null.

**Then route the candidate by whether it has already been planned.** Run:

  backlog task view <id> --json

Read the \`implementationPlan\` field. Judge it by content, not by the presence of a "planned" label — that label is applied inconsistently and a ticket can carry a full plan without it.

- If \`implementationPlan\` holds a real plan (substantive implementation steps, not empty and not a placeholder), it does NOT need re-planning. Send it straight to execution:
    backlog task edit <id> -s "Dev Ready"
  Report destination: "Dev Ready".

- If \`implementationPlan\` is empty, missing, or just a stub, it needs planning:
    backlog task edit <id> -s "Needs Plan"
  Report destination: "Needs Plan".

Either edit moves the ticket out of "To Do" or "Blocked" on its own, so there is no label to clean up.

Report via structured output:
- ticketId: the chosen ticket's ID, or null if neither pass found an eligible ticket
- destination: "Dev Ready", "Needs Plan", or null
- reason: a short explanation of the choice, including which pass it came from and why you judged the plan present or absent (or why nothing was eligible)`;
}

function executePrompt(ticketId) {
  return `You're working in the meethook repo at its root (no git submodules — commits happen directly here).

First run \`backlog instructions task-execution\` and \`backlog instructions task-finalization\` so you follow the current backlog procedure.

Then use the Skill tool to invoke "/backlog-execute ${ticketId}". This skill will claim the ticket, implement the work, mark acceptance criteria, add implementation notes/summary, set the ticket status to Done, and commit the result — all per its own instructions. If the skill determines the ticket is blocked by new/unforeseen work, it will park the ticket per its own blocked-revert procedure and exit without completing it.

If the ticket does turn out to be blocked, the parking has to be durable or this loop will keep re-selecting it. Two things must be true, both via the \`backlog\` CLI: its status is "Blocked", and every blocking ticket ID is recorded in its \`--depends-on\` dependencies. The dependencies are what lets a later run tell a still-blocked ticket from one that can resume, so a "Blocked" status with no recorded blockers strands the ticket. Verify both before you report back.

Never bypass a git hook to land a commit. Do not pass \`--no-verify\` to any git command. If a pre-commit or pre-push hook fails, fix what it is complaining about — a hook failure is a real finding, and the attribution hook in particular is mandatory.

All backlog state changes must go through the \`backlog\` CLI — never hand-edit task markdown files.

After the skill finishes, report via structured output:
- ticketId: "${ticketId}"
- outcome: "completed" if the ticket was finished and committed, "blocked-reverted" if it was reverted to To Do, or "error" if something went wrong
- summary: one sentence describing what happened`;
}

function planPrompt(ticketId) {
  return `You're working in the meethook repo at its root (no git submodules).

First run \`backlog instructions task-creation\` and \`backlog instructions task-execution\` so you follow the current backlog procedure for creating sub-tickets and recording plans.

Then use the Skill tool to invoke "/backlog-planner ${ticketId}". This skill researches the ticket, analyzes dependencies, may create sub-tickets for discrete work, and writes a detailed implementation plan.

All backlog state changes must go through the \`backlog\` CLI — never hand-edit task markdown files.

Before you begin, run \`backlog task view ${ticketId} --json\` and read its \`implementationPlan\`. If it already holds a substantive plan, do NOT re-plan it from scratch — the loop routes planned tickets straight to Dev Ready, so arriving here with a plan already present means either the plan is stale or the ticket was mis-routed. In that case, review the existing plan against the current state of the code, revise it only where the code has moved on, and say so in your summary.

Once planning is complete, set the ticket's status to "Dev Ready" and label it planned in the same edit:
  backlog task edit ${ticketId} -s "Dev Ready" --add-label planned

The label lets later steps recognize planned work cheaply from a task list; the plan body remains the source of truth.

Report via structured output:
- ticketId: "${ticketId}"
- outcome: "planned" if planning completed and the status was set to Dev Ready, or "error" if something went wrong
- summary: one sentence describing what was planned (and any sub-tickets created)`;
}

phase("Setup");
const setup = await agent(SETUP_PROMPT, {
  schema: SETUP_SCHEMA,
  model: "haiku",
  phase: "Setup",
});
if (!setup) {
  return {
    stopReason: "setup-error",
    iterations: 0,
    results: [],
    table: "(setup failed)",
  };
}
log(
  `Setup: statuses = [${setup.statuses.join(", ")}]${setup.changed ? " (updated config.yml)" : ""}`,
);

const results = [];
let stopReason = "cap";

// How many times each ticket has been executed and reverted as blocked in this run.
// The labels and dependencies the executor writes are the durable fix; this is the
// backstop for the case where a revert fails to record its blocker, which would
// otherwise let one ticket cycle Dev Ready -> To Do -> Dev Ready for the whole run.
const blockedReverts = new Map();
// Tickets this run has stopped selecting entirely, after a second blocked revert.
const parked = new Set();

function park(ticketId) {
  const count = (blockedReverts.get(ticketId) ?? 0) + 1;
  blockedReverts.set(ticketId, count);
  if (count >= 2) {
    parked.add(ticketId);
    log(
      `Parked ${ticketId} for the rest of this run: reverted as blocked ${count} times.`,
    );
  }
}

for (let i = 0; i < MAX_ITERATIONS; i++) {
  phase("State");
  const state = await agent(STATE_PROMPT, {
    schema: STATE_SCHEMA,
    model: "haiku",
    phase: "State",
  });
  if (!state) {
    stopReason = "state-error";
    log("State detection failed; stopping.");
    break;
  }

  const executable = [...state.inProgress, ...state.devReady].filter(
    (id) => !parked.has(id),
  );
  if (executable.length > 0) {
    const target = executable[0];
    phase("Execute");
    log(`Iteration ${i + 1}: execute -> ${target}`);
    const outcome = await agent(executePrompt(target), {
      schema: ACTION_SCHEMA,
      phase: "Execute",
    });
    if (!outcome) {
      results.push({
        ticketId: target,
        phase: "execute",
        outcome: "error",
        summary: "subagent returned no result",
      });
      stopReason = "execute-error";
      break;
    }
    results.push({
      ticketId: target,
      phase: "execute",
      outcome: outcome.outcome,
      summary: outcome.summary,
    });
    if (outcome.outcome === "blocked-reverted") park(target);
    continue;
  }

  const plannable = state.needsPlan.filter((id) => !parked.has(id));
  if (plannable.length > 0) {
    const target = plannable[0];
    phase("Plan");
    log(`Iteration ${i + 1}: plan -> ${target}`);
    const outcome = await agent(planPrompt(target), {
      schema: ACTION_SCHEMA,
      phase: "Plan",
    });
    if (!outcome) {
      results.push({
        ticketId: target,
        phase: "plan",
        outcome: "error",
        summary: "subagent returned no result",
      });
      stopReason = "plan-error";
      break;
    }
    results.push({
      ticketId: target,
      phase: "plan",
      outcome: outcome.outcome,
      summary: outcome.summary,
    });
    continue;
  }

  phase("Choose");
  const choice = await agent(choosePrompt([...parked]), {
    schema: CHOOSE_SCHEMA,
    model: "haiku",
    phase: "Choose",
  });
  if (!choice) {
    stopReason = "choose-error";
    log("Choose step failed; stopping.");
    break;
  }
  if (!choice.ticketId) {
    stopReason = parked.size > 0 ? "drained-with-parked" : "drained";
    log(`Backlog drained: ${choice.reason}`);
    break;
  }
  // A ticket that arrives here already carrying a plan goes straight to Dev Ready;
  // re-planning it is the churn this loop used to spend whole iterations on.
  const queuedFor =
    choice.destination === "Dev Ready"
      ? "queued-for-execution"
      : "queued-for-planning";
  log(
    `Iteration ${i + 1}: choose -> ${choice.ticketId} (${choice.destination ?? "Needs Plan"})`,
  );
  results.push({
    ticketId: choice.ticketId,
    phase: "choose",
    outcome: queuedFor,
    summary: choice.reason,
  });
}

const table = [
  "| Ticket | Phase | Outcome | Summary |",
  "|---|---|---|---|",
  ...results.map(
    (r) => `| ${r.ticketId} | ${r.phase} | ${r.outcome} | ${r.summary} |`,
  ),
].join("\n");

if (parked.size > 0) {
  log(
    `Parked this run (reverted as blocked twice, not re-selected): ${[...parked].join(", ")}`,
  );
}

return {
  stopReason,
  iterations: results.length,
  parked: [...parked],
  results,
  table,
};
