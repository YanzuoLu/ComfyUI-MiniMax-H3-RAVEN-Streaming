/**
 * Display state for the preview widget.
 *
 * Pure module - no DOM, no ComfyUI imports, no import-time side effects.
 *
 * Three independent facts feed one label:
 *   - `backend`    what the sampler last reported (waiting / model_loading /
 *                  sampling / finalizing)
 *   - `media`      what the local MediaSource pipeline is doing (idle /
 *                  starting / playing / stalled / gap / failed)
 *   - `connection` the websocket (online / reconnecting)
 *
 * Keeping the resolution in one pure function is what makes the state machine
 * testable without a browser, and stops the label from being written from six
 * different callbacks.
 */

export const STATES = Object.freeze([
  'waiting',
  'model_loading',
  'sampling',
  'buffering',
  'live',
  'finalizing',
  'complete',
  'cancelled',
  'error',
  'reconnecting',
])

/**
 * Short, factual labels. No exclamation marks, no product voice.
 * `tone` maps to a CSS modifier class, not to a colour directly.
 */
export const STATE_INFO = Object.freeze({
  waiting: { label: 'Waiting', tone: 'idle', detail: 'No stream yet' },
  model_loading: { label: 'Loading model', tone: 'busy', detail: null },
  sampling: { label: 'Sampling', tone: 'busy', detail: 'No frames yet' },
  buffering: { label: 'Buffering', tone: 'busy', detail: null },
  live: { label: 'Live', tone: 'live', detail: null },
  finalizing: { label: 'Finalizing', tone: 'busy', detail: null },
  complete: { label: 'Complete', tone: 'done', detail: null },
  cancelled: { label: 'Cancelled', tone: 'idle', detail: null },
  error: { label: 'Preview error', tone: 'bad', detail: null },
  reconnecting: { label: 'Reconnecting', tone: 'warn', detail: null },
})

export const MEDIA_PHASES = Object.freeze([
  'idle',
  'starting',
  'playing',
  'stalled',
  'gap',
  'failed',
])

/**
 * @param {object} input
 * @param {string} input.backend     last backend phase
 * @param {string} input.media       local pipeline phase
 * @param {string} input.connection  'online' | 'reconnecting'
 * @param {string|null} input.ended  end reason, or null while running
 * @returns {string} one of STATES
 */
export function resolveState({
  backend = 'waiting',
  media = 'idle',
  connection = 'online',
  ended = null,
} = {}) {
  // A finished stream keeps its terminal label even if the socket drops after:
  // "Complete" then "Reconnecting" would be a lie about the media.
  if (ended === 'complete') return 'complete'
  if (ended === 'cancelled') return 'cancelled'
  if (ended === 'error') return 'error'

  if (media === 'failed') return 'error'
  if (connection === 'reconnecting') return 'reconnecting'

  if (backend === 'finalizing') return 'finalizing'
  if (backend === 'model_loading') return 'model_loading'

  if (media === 'playing') return 'live'
  if (media === 'starting' || media === 'stalled' || media === 'gap') {
    return 'buffering'
  }

  if (backend === 'sampling') return 'sampling'
  return 'waiting'
}

/** Label/tone/default detail for a state id, with a safe fallback. */
export function describeState(state) {
  return STATE_INFO[state] || STATE_INFO.waiting
}

/** States after which no further media is expected. */
export function isTerminal(state) {
  return state === 'complete' || state === 'cancelled' || state === 'error'
}
