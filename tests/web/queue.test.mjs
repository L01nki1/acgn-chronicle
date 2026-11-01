import test from "node:test";
import assert from "node:assert/strict";

import { createQueryQueue } from "../../src/acgn_rag/web/static/queue.mjs";

/** A promise a test resolves by hand, standing in for one slow fetch. */
function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

test("a second intent during an in-flight request is queued instead of discarded", () => {
  const queue = createQueryQueue();
  assert.equal(queue.begin(), true);
  const first = () => {};
  const second = () => {};
  queue.queue(first);
  queue.queue(second);
  assert.equal(queue.hasPending(), true);
  // The newest intent wins: replaying every click would fire requests the reader no longer wants.
  assert.equal(queue.finish(), second);
  assert.equal(queue.hasPending(), false);
  assert.equal(queue.finish(), null);
});

test("the request slot is exclusive and reusable", () => {
  const queue = createQueryQueue();
  assert.equal(queue.begin(), true);
  assert.equal(queue.begin(), false);
  assert.equal(queue.isBusy(), true);
  queue.finish();
  assert.equal(queue.isBusy(), false);
  assert.equal(queue.begin(), true);
});

test("a request raised while busy starts only after the response settles", async () => {
  const queue = createQueryQueue();
  const order = [];
  const gates = [];

  const run = async () => {
    if (!queue.begin()) {
      queue.queue(run);
      return;
    }
    // Each run gets its own fetch, so the assertions track the real start/end of each request.
    const gate = deferred();
    gates.push(gate);
    order.push("start");
    try {
      await gate.promise;
      order.push("end");
    } finally {
      const next = queue.finish();
      if (next) {
        next();
      }
    }
  };

  const inFlight = run();
  await Promise.resolve();
  assert.deepEqual(order, ["start"]);

  await run();
  assert.deepEqual(order, ["start"], "the queued intent must not start a second fetch");
  assert.equal(queue.hasPending(), true);

  gates[0].resolve();
  await inFlight;
  await Promise.resolve();
  assert.deepEqual(order, ["start", "end", "start"]);
});
