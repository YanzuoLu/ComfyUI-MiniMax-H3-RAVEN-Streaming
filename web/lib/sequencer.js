/**
 * Strict in-order delivery for stream events.
 *
 * Pure module - no DOM, no ComfyUI imports, no import-time side effects.
 *
 * Websockets deliver in order on one connection, but a session can outlive a
 * connection (reconnect, backend resend, a duplicate broadcast to a client that
 * is subscribed twice). The sampler must never be blamed for a preview glitch,
 * so this layer is deliberately explicit about every abnormal case rather than
 * silently guessing:
 *
 *   - `seq` below the next expected one  -> duplicate / replay, dropped
 *   - `seq` equal to the next expected   -> released immediately, then any
 *                                           contiguous run that was waiting
 *   - `seq` above the next expected      -> held, and the session is "in a gap"
 *   - more held items than `maxPending`  -> overflow: the gap is unrecoverable
 *                                           from buffering alone and the caller
 *                                           should ask for a resume
 */

export const PUSH_RELEASED = 'released'
export const PUSH_DUPLICATE = 'duplicate'
export const PUSH_HELD = 'held'
export const PUSH_OVERFLOW = 'overflow'

export class Sequencer {
  /**
   * @param {object} [options]
   * @param {number} [options.maxPending] how many out-of-order items to hold
   */
  constructor({ maxPending = 64 } = {}) {
    this.maxPending = maxPending
    this.reset()
  }

  reset(nextSeq = 0) {
    /** @type {Map<number, unknown>} */
    this.pending = new Map()
    this.nextSeq = nextSeq
    this.lastReleasedSeq = nextSeq - 1
    this.stats = {
      released: 0,
      duplicates: 0,
      held: 0,
      overflows: 0,
      invalid: 0,
    }
  }

  /** True while at least one item is waiting on a missing predecessor. */
  get inGap() {
    return this.pending.size > 0
  }

  /** The seq the stream is stuck on, or null when there is no gap. */
  get missingSeq() {
    return this.pending.size > 0 ? this.nextSeq : null
  }

  /**
   * The last seq that was handed to the consumer. A resume request uses this:
   * everything up to and including it is known-good.
   */
  get resumeFrom() {
    return this.lastReleasedSeq
  }

  /**
   * Offer one item.
   *
   * @param {number} seq
   * @param {unknown} item
   * @returns {{status: string, released: Array<{seq: number, item: unknown}>, missing: number|null}}
   */
  push(seq, item) {
    if (!Number.isInteger(seq) || seq < 0) {
      this.stats.invalid += 1
      return { status: PUSH_DUPLICATE, released: [], missing: this.missingSeq }
    }

    // A late join (the first message of a session arriving with seq > 0, e.g.
    // the tab connected mid-run) falls out of the rules below as a gap at the
    // baseline seq, so the caller asks for a resume instead of appending
    // fragments the decoder has no init segment for.
    if (seq < this.nextSeq) {
      this.stats.duplicates += 1
      return { status: PUSH_DUPLICATE, released: [], missing: this.missingSeq }
    }

    if (seq > this.nextSeq) {
      if (this.pending.has(seq)) {
        this.stats.duplicates += 1
        return { status: PUSH_DUPLICATE, released: [], missing: this.missingSeq }
      }
      if (this.pending.size >= this.maxPending) {
        this.stats.overflows += 1
        return { status: PUSH_OVERFLOW, released: [], missing: this.missingSeq }
      }
      this.pending.set(seq, item)
      this.stats.held += 1
      return { status: PUSH_HELD, released: [], missing: this.missingSeq }
    }

    const released = [{ seq, item }]
    this.nextSeq = seq + 1
    while (this.pending.has(this.nextSeq)) {
      released.push({ seq: this.nextSeq, item: this.pending.get(this.nextSeq) })
      this.pending.delete(this.nextSeq)
      this.nextSeq += 1
    }
    this.lastReleasedSeq = released[released.length - 1].seq
    this.stats.released += released.length
    return { status: PUSH_RELEASED, released, missing: this.missingSeq }
  }

  /**
   * Give up on the missing seq and release what is already held.
   * The caller decides whether the result is appendable: for fMP4 only a
   * keyframe fragment can follow a hole, which is why the returned items are
   * handed back rather than applied here.
   */
  skipGap() {
    if (this.pending.size === 0) return { released: [], skipped: null }
    const skipped = this.nextSeq
    const seqs = [...this.pending.keys()].sort((a, b) => a - b)
    this.nextSeq = seqs[0]
    const out = []
    while (this.pending.has(this.nextSeq)) {
      out.push({ seq: this.nextSeq, item: this.pending.get(this.nextSeq) })
      this.pending.delete(this.nextSeq)
      this.nextSeq += 1
    }
    if (out.length > 0) {
      this.lastReleasedSeq = out[out.length - 1].seq
      this.stats.released += out.length
    }
    return { released: out, skipped }
  }
}
