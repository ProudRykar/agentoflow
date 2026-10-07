/**
 * Run accounting, kept apart from the context meter.
 *
 * The meter answers "how big was this request". This answers "what
 * has the run spent", which is the question that decides whether to
 * press Stop.
 *
 * The one rule that shapes the whole module: a provider that reports
 * no usage is not the same as a provider that reported zero. Ollama
 * frequently returns nothing, and a total that silently stayed at
 * zero would read as "this run was free".
 */

export interface RunUsageView {
  /** Prompt tokens, provider-reported, across the run. */
  promptTokens: number
  /** Completion tokens, provider-reported, across the run. */
  completionTokens: number
  /** Responses received. */
  calls: number
  /** Of those, how many reported no counts. */
  callsWithoutUsage: number
  /** USD, or null when any call in the run went unpriced. */
  estimatedCost: number | null
}

/**
 * Accumulator behind RunUsageView.
 *
 * Cost needs a separate "was every call priced" flag rather than
 * leaning on a null sentinel: null is also the starting value, so
 * without the flag a run whose every call *was* priced stays null
 * forever.
 */
interface RunUsageAccumulator extends RunUsageView {
  costSum: number
  costComplete: boolean
}

export function emptyRunUsage(): RunUsageView {
  return {
    promptTokens: 0,
    completionTokens: 0,
    calls: 0,
    callsWithoutUsage: 0,
    estimatedCost: null,
  }
}

export function addRunUsage(
  current: RunUsageView,
  runKey: string,
  addition: {
    promptTokens: number | null
    completionTokens: number | null
    estimatedCost: number | null
  },
): RunUsageView {
  // Subagents bill their own work. Folding them into the main run's
  // total would show one number that no provider ever produced.
  if (runKey.includes(':')) {
    return current
  }

  const before = current as Partial<RunUsageAccumulator>

  // Defaults rather than required fields: the accumulator state has to
  // survive a store restored from an older snapshot, where these two
  // simply are not present yet.
  const costSum = (before.costSum ?? 0) + (addition.estimatedCost ?? 0)
  const base = current as RunUsageView

  // Sticky false. Prices come from configuration and so apply to the
  // whole run: one unpriced call means the total is unknown, and a
  // partial sum would understate the bill while still looking like a
  // bill.
  const costComplete =
    (before.costComplete ?? true) && addition.estimatedCost !== null

  return {
    promptTokens: base.promptTokens + (addition.promptTokens ?? 0),
    completionTokens:
      base.completionTokens + (addition.completionTokens ?? 0),
    calls: base.calls + 1,
    callsWithoutUsage:
      addition.promptTokens === null
        ? base.callsWithoutUsage + 1
        : base.callsWithoutUsage,
    estimatedCost: costComplete ? costSum : null,
    costSum,
    costComplete,
  } as RunUsageAccumulator
}

/** Whether anything at all was actually measured. */
export function isMeasured(usage: RunUsageView): boolean {
  return usage.calls > usage.callsWithoutUsage
}

export function totalTokens(usage: RunUsageView): number {
  return usage.promptTokens + usage.completionTokens
}
