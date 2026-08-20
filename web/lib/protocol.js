/**
 * RAVEN Streaming preview - wire protocol helpers.
 *
 * Pure module: no DOM, no ComfyUI imports, no side effects at import time.
 * It is safe for the ComfyUI `/extensions` glob to load this file directly
 * (it registers nothing), and it is runnable under plain Node for tests.
 *
 * The wire format itself is specified in `web/PROTOCOL.md`. Keep the two in
 * sync: the static tests compare the constants below against that document.
 */

/** Websocket message type the backend must pass to `PromptServer.send_sync`. */
export const MESSAGE_TYPE = 'raven.preview'

/** Protocol version understood by this client. */
export const PROTOCOL_VERSION = 1

/** Every value `envelope.event` may take. */
export const EVENT_KINDS = Object.freeze([
  'open',
  'init',
  'segment',
  'status',
  'end',
])

/** Phases the backend may report through an `status` event. */
export const BACKEND_PHASES = Object.freeze([
  'waiting',
  'model_loading',
  'sampling',
  'finalizing',
])

/** Reasons a stream may end. */
export const END_REASONS = Object.freeze(['complete', 'cancelled', 'error'])

/** Payload encodings the client accepts for `init` / `segment` bodies. */
export const PAYLOAD_ENCODINGS = Object.freeze(['base64'])

/** Optional HTTP route the client uses to ask for a replay after a gap. */
export const RESUME_ROUTE = '/raven_streaming/preview/resume'

const BASE64_RE = /^[A-Za-z0-9+/]*={0,2}$/

class ProtocolError extends Error {}

function fail(message) {
  throw new ProtocolError(message)
}

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function asString(value, field) {
  if (typeof value !== 'string' || value.length === 0) {
    fail(`${field} must be a non-empty string`)
  }
  return value
}

function asInt(value, field, { min = 0 } = {}) {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < min) {
    fail(`${field} must be an integer >= ${min}`)
  }
  return value
}

function asOptionalNumber(value, field) {
  if (value === undefined || value === null) return null
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    fail(`${field} must be a finite number`)
  }
  return value
}

function asOptionalString(value, field) {
  if (value === undefined || value === null) return null
  if (typeof value !== 'string') fail(`${field} must be a string`)
  return value
}

/**
 * Decode a base64 body into bytes.
 *
 * `atob` exists both in browsers and in Node >= 16, so this stays testable
 * outside a browser.
 */
export function decodeBase64(data, field = 'data') {
  if (typeof data !== 'string') fail(`${field} must be a base64 string`)
  const cleaned = data.replace(/\s+/g, '')
  if (!BASE64_RE.test(cleaned)) fail(`${field} is not valid base64`)
  let binary
  try {
    binary = atob(cleaned)
  } catch (err) {
    fail(`${field} could not be decoded: ${err && err.message}`)
  }
  const out = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i)
  return out
}

/**
 * Byte cost of carrying `n` payload bytes as a base64 JSON string.
 * Used by the docs and by the size guard below; kept here so there is one
 * definition rather than a number repeated in prose.
 */
export function base64Cost(n) {
  return Math.ceil(n / 3) * 4
}

/**
 * Validate and normalise one websocket message body.
 *
 * Returns a frozen, fully-typed envelope. Throws `ProtocolError` on anything
 * malformed; callers treat that as "drop this message and count it", never as
 * a reason to tear the preview down.
 */
export function parseEnvelope(raw) {
  if (!isPlainObject(raw)) fail('message body must be an object')

  const version = raw.v
  if (version !== PROTOCOL_VERSION) {
    fail(`unsupported protocol version ${JSON.stringify(version)}`)
  }

  const event = asString(raw.event, 'event')
  if (!EVENT_KINDS.includes(event)) fail(`unknown event kind ${event}`)

  const envelope = {
    v: version,
    event,
    sessionId: asString(raw.session_id, 'session_id'),
    nodeId: asString(raw.node_id, 'node_id'),
    promptId: asOptionalString(raw.prompt_id, 'prompt_id'),
    seq: asInt(raw.seq, 'seq'),
    t: asOptionalNumber(raw.t, 't'),
    body: null,
  }

  switch (event) {
    case 'open':
      envelope.body = parseOpen(raw)
      break
    case 'init':
      envelope.body = parsePayload(raw, 'init')
      break
    case 'segment':
      envelope.body = parseSegment(raw)
      break
    case 'status':
      envelope.body = parseStatus(raw)
      break
    case 'end':
      envelope.body = parseEnd(raw)
      break
    default:
      fail(`unhandled event kind ${event}`)
  }

  return Object.freeze(envelope)
}

function parseOpen(raw) {
  const mime = asString(raw.mime, 'mime')
  if (!/^video\/mp4\s*;/i.test(mime)) {
    fail('mime must be a fragmented MP4 type with a codecs parameter')
  }
  if (!/codecs\s*=/i.test(mime)) fail('mime is missing the codecs parameter')

  const audio = raw.audio === undefined || raw.audio === null ? null : raw.audio
  if (audio !== null && !isPlainObject(audio)) fail('audio must be an object or null')

  return Object.freeze({
    mime,
    width: raw.width === undefined ? null : asInt(raw.width, 'width', { min: 1 }),
    height: raw.height === undefined ? null : asInt(raw.height, 'height', { min: 1 }),
    fps: asOptionalNumber(raw.fps, 'fps'),
    durationHint: asOptionalNumber(raw.duration_hint, 'duration_hint'),
    audio: audio
      ? Object.freeze({
          sampleRate: asOptionalNumber(audio.sample_rate, 'audio.sample_rate'),
          channels: asOptionalNumber(audio.channels, 'audio.channels'),
        })
      : null,
    resync: raw.resync === true,
    label: asOptionalString(raw.label, 'label'),
  })
}

function parsePayload(raw, what) {
  const encoding = asString(raw.encoding, 'encoding')
  if (!PAYLOAD_ENCODINGS.includes(encoding)) {
    fail(`unsupported ${what} encoding ${encoding}`)
  }
  const declared = raw.bytes === undefined ? null : asInt(raw.bytes, 'bytes', { min: 1 })
  const bytes = decodeBase64(raw.data, `${what}.data`)
  if (declared !== null && declared !== bytes.length) {
    fail(`${what} length mismatch: declared ${declared}, decoded ${bytes.length}`)
  }
  return Object.freeze({ encoding, bytes })
}

function parseSegment(raw) {
  const payload = parsePayload(raw, 'segment')
  return Object.freeze({
    encoding: payload.encoding,
    bytes: payload.bytes,
    index: raw.index === undefined ? null : asInt(raw.index, 'index'),
    keyframe: raw.keyframe === true,
    start: asOptionalNumber(raw.start, 'start'),
    duration: asOptionalNumber(raw.duration, 'duration'),
  })
}

function parseStatus(raw) {
  const phase = asString(raw.phase, 'phase')
  if (!BACKEND_PHASES.includes(phase)) fail(`unknown phase ${phase}`)
  let progress = null
  if (raw.progress !== undefined && raw.progress !== null) {
    if (!isPlainObject(raw.progress)) fail('progress must be an object')
    progress = Object.freeze({
      value: asOptionalNumber(raw.progress.value, 'progress.value'),
      max: asOptionalNumber(raw.progress.max, 'progress.max'),
    })
  }
  return Object.freeze({
    phase,
    message: asOptionalString(raw.message, 'message'),
    progress,
  })
}

function parseEnd(raw) {
  const reason = asString(raw.reason, 'reason')
  if (!END_REASONS.includes(reason)) fail(`unknown end reason ${reason}`)
  return Object.freeze({
    reason,
    message: asOptionalString(raw.message, 'message'),
    /** Total number of media segments the backend claims to have sent. */
    segments: raw.segments === undefined ? null : asInt(raw.segments, 'segments'),
  })
}

export { ProtocolError }
