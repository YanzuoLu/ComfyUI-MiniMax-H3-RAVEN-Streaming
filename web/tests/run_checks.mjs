/**
 * Behavioural checks for the browser-free parts of the preview extension.
 *
 * Run by `tests/test_web_behaviour.py` through Node; also runnable by hand:
 *
 *     node web/tests/run_checks.mjs
 *
 * The `.mjs` extension keeps this file out of ComfyUI's `/extensions` glob,
 * which only collects `*.js`, so nothing here is ever shipped to a browser.
 * The directory name also matches the `tests/` rule in `.comfyignore`, so it
 * is excluded from the published package.
 *
 * What is covered: envelope parsing, sequence ordering / de-duplication / gap
 * handling, the state resolver, node and node-type matching, and the
 * MediaSource append queue driven through fakes. What is not covered: the DOM
 * widget and real playback - those need a browser.
 */

import { strict as assert } from 'node:assert'

import {
  MESSAGE_TYPE,
  PROTOCOL_VERSION,
  parseEnvelope,
  decodeBase64,
  base64Cost,
  ProtocolError,
} from '../lib/protocol.js'
import {
  Sequencer,
  PUSH_RELEASED,
  PUSH_DUPLICATE,
  PUSH_HELD,
  PUSH_OVERFLOW,
} from '../lib/sequencer.js'
import { resolveState, isTerminal } from '../lib/states.js'
import { isSamplerNode, matchesNode, routeToController } from '../lib/identity.js'
import { MediaPipeline } from '../lib/mse.js'
import { FakeVideoElement, makeMediaEnv } from './fakes.mjs'

const tests = []
const test = (name, fn) => tests.push([name, fn])

/* ------------------------------------------------------------- fixtures */

const b64 = (bytes) => Buffer.from(bytes).toString('base64')

const openEvent = (over = {}) => ({
  v: 1,
  event: 'open',
  session_id: 's1',
  node_id: '7',
  seq: 0,
  mime: 'video/mp4; codecs="avc1.640028,mp4a.40.2"',
  width: 848,
  height: 480,
  fps: 24,
  ...over,
})

const segmentEvent = (seq, over = {}) => ({
  v: 1,
  event: 'segment',
  session_id: 's1',
  node_id: '7',
  seq,
  encoding: 'base64',
  data: b64([1, 2, 3, 4]),
  bytes: 4,
  ...over,
})

/* ------------------------------------------------------------- protocol */

test('message type and version are the documented ones', () => {
  assert.equal(MESSAGE_TYPE, 'raven.preview')
  assert.equal(PROTOCOL_VERSION, 1)
})

test('a well-formed open event parses', () => {
  const env = parseEnvelope(openEvent())
  assert.equal(env.event, 'open')
  assert.equal(env.sessionId, 's1')
  assert.equal(env.nodeId, '7')
  assert.equal(env.seq, 0)
  assert.equal(env.body.width, 848)
  assert.equal(env.body.resync, false)
})

test('an open event without a codecs parameter is rejected', () => {
  assert.throws(
    () => parseEnvelope(openEvent({ mime: 'video/mp4' })),
    ProtocolError,
  )
})

test('a foreign protocol version is rejected', () => {
  assert.throws(() => parseEnvelope(openEvent({ v: 2 })), ProtocolError)
})

test('segment payloads decode and length is verified', () => {
  const env = parseEnvelope(segmentEvent(4))
  assert.deepEqual([...env.body.bytes], [1, 2, 3, 4])
  assert.throws(() => parseEnvelope(segmentEvent(4, { bytes: 9 })), ProtocolError)
})

test('non-base64 payloads are rejected rather than decoded to garbage', () => {
  assert.throws(() => parseEnvelope(segmentEvent(4, { data: '!!!!' })), ProtocolError)
  assert.throws(() => decodeBase64(42), ProtocolError)
})

test('unknown event kinds, phases and end reasons are rejected', () => {
  assert.throws(() => parseEnvelope(openEvent({ event: 'nope' })), ProtocolError)
  assert.throws(
    () =>
      parseEnvelope({
        v: 1,
        event: 'status',
        session_id: 's1',
        node_id: '7',
        seq: 1,
        phase: 'live',
      }),
    ProtocolError,
  )
  assert.throws(
    () =>
      parseEnvelope({
        v: 1,
        event: 'end',
        session_id: 's1',
        node_id: '7',
        seq: 2,
        reason: 'stopped',
      }),
    ProtocolError,
  )
})

test('base64 cost matches the figure quoted in PROTOCOL.md', () => {
  assert.equal(base64Cost(3), 4)
  assert.equal(base64Cost(200 * 1024), 273068)
  assert.ok(base64Cost(1_000_000) / 1_000_000 < 1.34)
})

/* ------------------------------------------------------------ sequencer */

test('in-order pushes release immediately', () => {
  const seq = new Sequencer()
  for (let i = 0; i < 5; i++) {
    const result = seq.push(i, `item${i}`)
    assert.equal(result.status, PUSH_RELEASED)
    assert.equal(result.released.length, 1)
  }
  assert.equal(seq.inGap, false)
  assert.equal(seq.resumeFrom, 4)
})

test('out-of-order pushes are held and released as a run', () => {
  const seq = new Sequencer()
  seq.push(0, 'a')
  assert.equal(seq.push(2, 'c').status, PUSH_HELD)
  assert.equal(seq.push(3, 'd').status, PUSH_HELD)
  assert.equal(seq.inGap, true)
  assert.equal(seq.missingSeq, 1)

  const result = seq.push(1, 'b')
  assert.equal(result.status, PUSH_RELEASED)
  assert.deepEqual(
    result.released.map((r) => r.item),
    ['b', 'c', 'd'],
  )
  assert.equal(seq.inGap, false)
  assert.equal(seq.resumeFrom, 3)
})

test('replayed and repeated packets are dropped, not appended twice', () => {
  const seq = new Sequencer()
  seq.push(0, 'a')
  seq.push(1, 'b')
  assert.equal(seq.push(1, 'b-again').status, PUSH_DUPLICATE)
  assert.equal(seq.push(0, 'a-again').status, PUSH_DUPLICATE)
  seq.push(3, 'd')
  assert.equal(seq.push(3, 'd-again').status, PUSH_DUPLICATE)
  assert.equal(seq.stats.duplicates, 3)
  assert.equal(seq.stats.released, 2)
})

test('the reorder backlog is bounded', () => {
  const seq = new Sequencer({ maxPending: 3 })
  seq.push(0, 'a')
  seq.push(2, 'c')
  seq.push(3, 'd')
  seq.push(4, 'e')
  assert.equal(seq.push(5, 'f').status, PUSH_OVERFLOW)
  assert.equal(seq.missingSeq, 1)
})

test('a late join reads as a gap at the baseline', () => {
  const seq = new Sequencer()
  assert.equal(seq.push(42, 'mid-stream').status, PUSH_HELD)
  assert.equal(seq.missingSeq, 0)
  assert.equal(seq.resumeFrom, -1)
})

test('skipGap gives back what was held', () => {
  const seq = new Sequencer()
  seq.push(0, 'a')
  seq.push(2, 'c')
  seq.push(3, 'd')
  const { released, skipped } = seq.skipGap()
  assert.equal(skipped, 1)
  assert.deepEqual(
    released.map((r) => r.item),
    ['c', 'd'],
  )
  assert.equal(seq.inGap, false)
})

test('a reset drops every held packet of the old session', () => {
  const seq = new Sequencer()
  seq.push(0, 'a')
  seq.push(5, 'held')
  seq.reset(0)
  assert.equal(seq.inGap, false)
  assert.equal(seq.push(0, 'new').status, PUSH_RELEASED)
})

/* --------------------------------------------------------------- states */

test('media phase and backend phase combine into one label', () => {
  assert.equal(resolveState({}), 'waiting')
  assert.equal(resolveState({ backend: 'model_loading' }), 'model_loading')
  assert.equal(resolveState({ backend: 'sampling' }), 'sampling')
  assert.equal(resolveState({ backend: 'sampling', media: 'playing' }), 'live')
  assert.equal(resolveState({ backend: 'sampling', media: 'starting' }), 'buffering')
  assert.equal(resolveState({ backend: 'sampling', media: 'gap' }), 'buffering')
  assert.equal(resolveState({ backend: 'sampling', media: 'stalled' }), 'buffering')
  assert.equal(resolveState({ backend: 'finalizing', media: 'playing' }), 'finalizing')
})

test('a dropped socket outranks the media phase, a finished stream does not', () => {
  assert.equal(
    resolveState({ backend: 'sampling', media: 'playing', connection: 'reconnecting' }),
    'reconnecting',
  )
  assert.equal(
    resolveState({ connection: 'reconnecting', ended: 'complete' }),
    'complete',
  )
})

test('terminal states win over everything', () => {
  assert.equal(resolveState({ backend: 'sampling', ended: 'cancelled' }), 'cancelled')
  assert.equal(resolveState({ media: 'playing', ended: 'error' }), 'error')
  assert.equal(resolveState({ media: 'failed' }), 'error')
  assert.ok(isTerminal('complete') && isTerminal('cancelled') && isTerminal('error'))
  assert.ok(!isTerminal('live') && !isTerminal('buffering'))
})

/* ------------------------------------------------------------- identity */

test('the sampler node is matched by mapping key or display name', () => {
  assert.ok(isSamplerNode('RAVENStreamingSampler'))
  assert.ok(isSamplerNode('RavenStreamingSampler'))
  assert.ok(isSamplerNode('SomeKey', 'RAVEN Streaming Sampler'))
  assert.ok(isSamplerNode('raven_streaming_sampler'))
  assert.ok(!isSamplerNode('RAVENModelLoader'))
  assert.ok(!isSamplerNode('KSampler'))
  assert.ok(!isSamplerNode(undefined, undefined))
})

test('node ids match exactly, or by subgraph path suffix', () => {
  assert.ok(matchesNode(7, '7'))
  assert.ok(matchesNode('7', '12:7'))
  assert.ok(!matchesNode(7, '17'))
  assert.ok(!matchesNode(-1, '-1'))
  assert.ok(!matchesNode(7, ''))
})

test('an exact node id wins over a subgraph suffix', () => {
  const a = { nodeId: 3 }
  const b = { nodeId: '12:3' }
  assert.equal(routeToController([a, b], '12:3'), b)
  assert.equal(routeToController([a, b], '3'), a)
  assert.equal(routeToController([a, b], '99'), null)
})

test('two nodes never share a stream', () => {
  const seven = { nodeId: 7 }
  const eight = { nodeId: 8 }
  assert.equal(routeToController([seven, eight], '8'), eight)
  assert.equal(routeToController([seven, eight], '7'), seven)
})

/* ------------------------------------------------------------ mse queue */

function makePipeline(overrides = {}) {
  const { env, created, revoked } = makeMediaEnv()
  const video = new FakeVideoElement()
  const errors = []
  const phases = []
  const pipeline = new MediaPipeline({
    video,
    env,
    onError: (message) => errors.push(message),
    onPhase: (phase) => phases.push(phase),
    ...overrides,
  })
  return { pipeline, video, env, created, revoked, errors, phases }
}

const MIME = 'video/mp4; codecs="avc1.640028,mp4a.40.2"'

test('an unsupported mime type is refused without throwing', () => {
  const { pipeline, errors, phases } = makePipeline()
  assert.equal(pipeline.open('video/webm; codecs="vp9"'), false)
  assert.ok(errors[0].includes('cannot decode'))
  assert.ok(phases.includes('failed'))
})

test('init and fragments append one at a time, in order', () => {
  const { pipeline, video } = makePipeline()
  assert.equal(pipeline.open(MIME), true)
  const source = pipeline.mediaSource
  source.openNow()

  pipeline.appendInit(Uint8Array.from([1]))
  pipeline.appendSegment(Uint8Array.from([2]))
  pipeline.appendSegment(Uint8Array.from([3]))

  const sb = source.sourceBuffers[0]
  assert.equal(sb.appended.length, 1, 'only one append is in flight at a time')
  source.settle()
  source.settle()
  assert.deepEqual(
    sb.appended.map((chunk) => chunk[0]),
    [1, 2, 3],
  )
  assert.equal(video.src, 'blob:fake/1')
})

test('a fragment arriving before the init segment is refused', () => {
  const { pipeline, errors } = makePipeline()
  pipeline.open(MIME)
  pipeline.mediaSource.openNow()
  pipeline.appendSegment(Uint8Array.from([2]))
  assert.equal(pipeline.mediaSource.sourceBuffers[0].appended.length, 0)
  assert.ok(errors.some((message) => message.includes('before the init segment')))
})

test('a second init segment for one source is refused', () => {
  const { pipeline, errors } = makePipeline()
  pipeline.open(MIME)
  pipeline.mediaSource.openNow()
  pipeline.appendInit(Uint8Array.from([1]))
  pipeline.appendInit(Uint8Array.from([9]))
  assert.ok(errors.some((message) => message.includes('second init segment')))
})

test('a full buffer is trimmed behind the playhead and the append retried', () => {
  const { pipeline } = makePipeline()
  pipeline.open(MIME)
  pipeline.mediaSource.openNow()
  const sb = pipeline.mediaSource.sourceBuffers[0]
  sb.quotaUntil = 1
  pipeline.video.currentTime = 20

  pipeline.appendInit(Uint8Array.from([1]))
  assert.equal(sb.removed.length, 1, 'eviction ran')
  assert.deepEqual(sb.removed[0], [0, 14])
  pipeline.mediaSource.settle()
  assert.equal(sb.appended.length, 1, 'the append was retried after eviction')
})

test('endOfStream waits for the queue to drain', () => {
  const { pipeline } = makePipeline()
  pipeline.open(MIME)
  const source = pipeline.mediaSource
  source.openNow()
  pipeline.appendInit(Uint8Array.from([1]))
  pipeline.appendSegment(Uint8Array.from([2]))
  pipeline.endOfStream()
  assert.equal(source.endedCount, 0, 'not while a fragment is still queued')
  source.settle()
  source.settle()
  assert.equal(source.endedCount, 1)
  assert.equal(pipeline.ended, true)
})

test('close stops the queue, revokes the URL and removes every listener', () => {
  const { pipeline, video, revoked } = makePipeline()
  pipeline.open(MIME)
  const source = pipeline.mediaSource
  source.openNow()
  pipeline.appendInit(Uint8Array.from([1]))
  pipeline.appendSegment(Uint8Array.from([2]))
  const sb = source.sourceBuffers[0]

  pipeline.close()

  assert.deepEqual(revoked, ['blob:fake/1'])
  assert.equal(pipeline.queue.length, 0)
  assert.equal(video.src, '')
  assert.equal(video.loadCount, 1)
  assert.equal(video.paused, true)
  assert.equal(video.listenerCount(), 0, 'video listeners removed')
  assert.equal(sb.listenerCount(), 0, 'source buffer listeners removed')
  assert.equal(source.listenerCount(), 0, 'media source listeners removed')
})

test('re-opening replaces the source and revokes the previous URL', () => {
  const { pipeline, revoked } = makePipeline()
  pipeline.open(MIME)
  pipeline.mediaSource.openNow()
  pipeline.appendInit(Uint8Array.from([1]))
  pipeline.open(MIME)
  assert.deepEqual(revoked, ['blob:fake/1'])
  assert.equal(pipeline.initAppended, false)
  assert.equal(pipeline.queue.length, 0)
})

test('destroy is idempotent and releases the element', () => {
  const { pipeline, revoked } = makePipeline()
  pipeline.open(MIME)
  pipeline.destroy()
  pipeline.destroy()
  assert.equal(pipeline.video, null)
  assert.equal(revoked.length, 1)
})

test('live playback catches up, but not after the viewer takes over', () => {
  const { pipeline, video } = makePipeline()
  pipeline.open(MIME)
  pipeline.mediaSource.openNow()
  pipeline.appendInit(Uint8Array.from([1]))
  pipeline.mediaSource.settle()
  assert.ok(video.currentTime > 28, 'jumped close to the buffer edge')

  video.currentTime = 0
  pipeline.userControlled = true
  pipeline.appendSegment(Uint8Array.from([2]))
  pipeline.mediaSource.settle()
  assert.equal(video.currentTime, 0, 'the viewer keeps their playhead')
})

/* ----------------------------------------------------------------- run */

let failed = 0
for (const [name, fn] of tests) {
  try {
    fn()
    console.log(`ok   ${name}`)
  } catch (err) {
    failed += 1
    console.log(`FAIL ${name}\n     ${err && err.message}`)
  }
}
console.log(`\n${tests.length - failed}/${tests.length} checks passed`)
process.exit(failed === 0 ? 0 : 1)
