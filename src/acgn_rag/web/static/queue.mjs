/**
 * One in-flight request at a time, with the latest competing intent queued instead of dropped.
 *
 * DOM-free on purpose: the page's coordination rule is testable without a browser, and the
 * regression that motivated it — an example-question click during a fetch that silently did
 * nothing while the previous response kept rendering — stays covered.
 */
export function createQueryQueue() {
  let busy = false;
  let pending = null;

  return {
    /** Claim the single request slot; false means a request is already running. */
    begin() {
      if (busy) {
        return false;
      }
      busy = true;
      return true;
    },
    /** Remember what to run next. The newest intent wins: replays are not useful to a reader. */
    queue(action) {
      pending = action;
    },
    /** Release the slot and hand back the queued action, if any. */
    finish() {
      busy = false;
      const next = pending;
      pending = null;
      return next;
    },
    isBusy() {
      return busy;
    },
    hasPending() {
      return pending !== null;
    },
  };
}
