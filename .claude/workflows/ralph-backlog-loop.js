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
      detail: "pick next unblocked To Do ticket and move it to Needs Plan",
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
        "The ticket ID moved to Needs Plan, or null if no eligible To Do ticket exists",
    },
    reason: {
      type: "string",
      description: "Short explanation of the choice or why nothing was chosen",
    },
  },
  required: ["ticketId", "reason"],
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

This project's backlog work loop needs a 5-stage pipeline:
  To Do -> Needs Plan -> Dev Ready -> In Progress -> Done

Run: backlog config get statuses

If "Dev Ready" is already in that list, change nothing.

If it is missing, edit backlog/config.yml and insert "Dev Ready" into the \`statuses\` list between "Needs Plan" and "In Progress", then save. (\`backlog config set\` does not accept the \`statuses\` key, so a direct file edit is the only way to change it — this is the one backlog file you may edit by hand. Never hand-edit task, draft, document, decision, or milestone markdown.) Do not remove, rename, or reorder any other existing statuses.

Report via structured output:
- statuses: the final statuses list (after your edit, if any)
- changed: true if you modified the file, false if it already had everything needed`;

const STATE_PROMPT = `In the meethook repo (repo root, no git submodules), run exactly one command:

  backlog task list --exclude-status "Backlog,To Do,Done" --json

That returns versioned JSON of shape {schemaVersion, kind: "task-list", tasks: [{id, title, status, labels, ordinal, ...}]}. Do not add --plain; --json and --plain cannot be combined.

Group the returned tasks by their \`status\` field and report the ticket IDs for each of "In Progress", "Dev Ready", and "Needs Plan", preserving the order they appear in the \`tasks\` array. Any status outside those three should be ignored.`;

const CHOOSE_PROMPT = `You're working in the meethook repo at its root.

Run exactly one discovery command:

  backlog task list -s "To Do" --ready --sort ordinal --json

\`--ready\` already restricts the result to unblocked tasks whose dependencies are all Done, and \`--sort ordinal\` returns them in the backlog's intended work order, so you do NOT need to inspect dependencies or run \`backlog task view\` on each candidate. Do not add --plain; --json and --plain cannot be combined.

The result is {schemaVersion, kind: "task-list", tasks: [{id, title, status, labels, ordinal, ...}]}.

Walk \`tasks\` in the order given and pick the first one whose \`labels\` array does NOT contain "no-ralph". If you find one, set its status to Needs Plan:
  backlog task edit <id> -s "Needs Plan"

Report via structured output:
- ticketId: the chosen ticket's ID, or null if the list was empty or every entry was labeled "no-ralph"
- reason: a short explanation of the choice (or why nothing was eligible)`;

function executePrompt(ticketId) {
  return `You're working in the meethook repo at its root (no git submodules — commits happen directly here).

First run \`backlog instructions task-execution\` and \`backlog instructions task-finalization\` so you follow the current backlog procedure.

Then use the Skill tool to invoke "/backlog-execute ${ticketId}". This skill will claim the ticket, implement the work, mark acceptance criteria, add implementation notes/summary, set the ticket status to Done, and commit the result — all per its own instructions. If the skill determines the ticket is blocked by new/unforeseen work, it will revert the ticket's status to "To Do" and exit without completing it.

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

Once planning is complete, set the ticket's status to "Dev Ready":
  backlog task edit ${ticketId} -s "Dev Ready"

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

  if (state.inProgress.length > 0 || state.devReady.length > 0) {
    const target = state.inProgress[0] || state.devReady[0];
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
    continue;
  }

  if (state.needsPlan.length > 0) {
    const target = state.needsPlan[0];
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
  const choice = await agent(CHOOSE_PROMPT, {
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
    stopReason = "drained";
    log(`Backlog drained: ${choice.reason}`);
    break;
  }
  log(`Iteration ${i + 1}: choose -> ${choice.ticketId}`);
  results.push({
    ticketId: choice.ticketId,
    phase: "choose",
    outcome: "queued-for-planning",
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

return { stopReason, iterations: results.length, results, table };
