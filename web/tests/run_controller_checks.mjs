/**
 * Controller-level checks: session isolation, ordering, cancellation, resume
 * and teardown, driven by mock websocket events.
 *
 * Run by `tests/test_web_behaviour.py`; also runnable by hand:
 *
 *     node web/tests/run_controller_checks.mjs
 *
 * The DOM here is the small fake in ./fakes.mjs. It exercises the logic that
 * decides *what* to show and *when* to release resources - not layout, styling
 * or real playback, which need a browser.
 */

import { strict as assert } from 'node:assert'

import {
  installFakeDom,
  installFakeTimers,
  makeMediaEnv,
  FakeVideoElement,
} from './fakes.mjs'

const dom = installFakeDom()

const { PreviewController, CONTROLLER_TUNING } = await import('../lib/controller.js')

const MIME = 'video/mp4; codecs="avc1.640028,mp4a.40.2"'
const b64 = (bytes) => Buffer.from(bytes).toString('base64')

const tests = []
const test = (name, fn) => tests.push([name, fn])

/** Let queued microtasks run; resume requests are promise-based. */
const flush = () => new Promise((resolve) => setImmediate(resolve))

function makeController({ requestResume } = {}) {
  const { env, revoked } = makeMediaEnv()
  const logs = []
  const resumeCalls = []
  const controller = new PreviewController({
    node: { id: 7, graph: {} },
    env,
    log: (message, err) => logs.push([message, err]),
    requestResume:
      requestResume ||
      ((request) => {
        resumeCalls.push(request)
        return Promise.resolve({ supported: true })
      }),
  })
  return { controller, revoked, logs, resumeCalls }
}

/** Everything a test needs about the live pipeline. */
const source = (controller) => controller.pipeline.mediaSource
const buffer = (controller) => source(controller).sourceBuffers[0]
const state = (controller) => controller.ui.root.dataset.state

const openMsg = (over = {}) => ({
  v: 1,
  event: 'open',
  session_id: 's1',
  node_id: '7',
  seq: 0,
  mime: MIME,
  width: 848,
  height: 480,
  ...over,
})

const initMsg = (seq = 1, over = {}) => ({
  v: 1,
  event: 'init',
  session_id: 's1',
  node_id: '7',
  seq,
  encoding: 'base64',
  data: b64([0xff]),
  ...over,
})

const segMsg = (seq, byte, over = {}) => ({
  v: 1,
  event: 'segment',
  session_id: 's1',
  node_id: '7',
  seq,
  encoding: 'base64',
  data: b64([byte]),
  ...over,
})

const statusMsg = (seq, phase, over = {}) => ({
  v: 1,
  event: 'status',
  session_id: 's1',
  node_id: '7',
  seq,
  phase,
  ...over,
})

const endMsg = (seq, reason, over = {}) => ({
  v: 1,
  event: 'end',
  session_id: 's1',
  node_id: '7',
  seq,
  reason,
  ...over,
})

/** open + init + one keyframe fragment, all settled. */
function startStream(controller) {
  controller.handleMessage(openMsg())
  source(controller).openNow()
  controller.handleMessage(initMsg(1))
  controller.handleMessage(segMsg(2, 0x01, { keyframe: true }))
  source(controller).settle(3)
}

/* --------------------------------------------------------------- basics */

test('a stream opens, appends in order and reports live once playing', () => {
  const { controller } = makeController()
  assert.equal(state(controller), 'waiting')

  startStream(controller)
  assert.deepEqual(
    buffer(controller).appended.map((chunk) => chunk[0]),
    [0xff, 0x01],
  )
  assert.equal(state(controller), 'buffering')

  controller.handleMessage(statusMsg(3, 'sampling'))
  controller.ui.video.emit('playing')
  assert.equal(state(controller), 'live')
  controller.destroy()
})

test('backend phases show up before any media arrives', () => {
  const { controller } = makeController()
  controller.handleMessage(openMsg())
  source(controller).openNow()
  controller.handleMessage(statusMsg(1, 'model_loading'))
  assert.equal(state(controller), 'model_loading')
  controller.handleMessage(statusMsg(2, 'finalizing'))
  assert.equal(state(controller), 'finalizing')
  controller.destroy()
})

test('the aspect ratio follows the announced frame size', () => {
  const { controller } = makeController()
  controller.handleMessage(openMsg({ width: 640, height: 480 }))
  assert.equal(controller.ui.root.style.getPropertyValue('--rvp-aspect'), '640 / 480')
  controller.destroy()
})

/* ------------------------------------------------------- order and loss */

test('out-of-order fragments are appended in sequence order', () => {
  const { controller } = makeController()
  startStream(controller)

  controller.handleMessage(segMsg(4, 0x04))
  source(controller).settle(2)
  assert.deepEqual(
    buffer(controller).appended.map((chunk) => chunk[0]),
    [0xff, 0x01],
    'seq 4 is held while seq 3 is missing',
  )
  assert.equal(state(controller), 'buffering')
  assert.ok(controller.ui.message.textContent.includes('segment 3'))

  controller.handleMessage(segMsg(3, 0x03))
  source(controller).settle(3)
  assert.deepEqual(
    buffer(controller).appended.map((chunk) => chunk[0]),
    [0xff, 0x01, 0x03, 0x04],
  )
  controller.destroy()
})

test('a repeated packet is dropped and counted, never appended twice', () => {
  const { controller } = makeController()
  startStream(controller)
  controller.handleMessage(segMsg(2, 0x01, { keyframe: true }))
  source(controller).settle(2)
  assert.equal(buffer(controller).appended.length, 2)
  assert.ok(controller.ui.meta.textContent.includes('1 dropped'))
  controller.destroy()
})

test('a malformed message is dropped, counted, and does not throw', () => {
  const { controller, logs } = makeController()
  startStream(controller)
  controller.handleMessage({ v: 1, event: 'segment', session_id: 's1' })
  controller.handleMessage(null)
  controller.handleMessage({ v: 99, event: 'open' })
  assert.equal(buffer(controller).appended.length, 2, 'nothing extra appended')
  assert.ok(logs.some(([message]) => message.includes('malformed')))
  controller.destroy()
})

test('a gap asks the backend to resume from the last good seq', async () => {
  const timers = installFakeTimers()
  try {
    const { controller, resumeCalls } = makeController()
    startStream(controller)
    controller.handleMessage(segMsg(5, 0x05))
    assert.equal(resumeCalls.length, 0, 'not before the timeout')
    timers.advance(CONTROLLER_TUNING.GAP_TIMEOUT_MS)
    await flush()
    assert.equal(resumeCalls.length, 1)
    assert.deepEqual(resumeCalls[0].nodeId, '7')
    assert.equal(resumeCalls[0].lastSeq, 2)
    controller.destroy()
  } finally {
    timers.uninstall()
  }
})

test('a backend without a resume route is only asked once', async () => {
  const timers = installFakeTimers()
  try {
    let calls = 0
    const { controller } = makeController({
      requestResume: () => {
        calls += 1
        return Promise.resolve({ supported: false })
      },
    })
    startStream(controller)
    controller.handleMessage(segMsg(5, 0x05))
    timers.advance(CONTROLLER_TUNING.GAP_TIMEOUT_MS)
    await flush()
    assert.equal(calls, 1)
    assert.equal(controller.resumeUnsupported, true)
    assert.ok(controller.ui.message.textContent.includes('does not support resume'))
    controller.destroy()
  } finally {
    timers.uninstall()
  }
})

/* ------------------------------------------------------------- sessions */

test('a message from another session is ignored', () => {
  const { controller } = makeController()
  startStream(controller)
  controller.handleMessage(segMsg(3, 0x09, { session_id: 'other' }))
  source(controller).settle(2)
  assert.equal(buffer(controller).appended.length, 2)
  controller.destroy()
})

test('a new session replaces the old one and releases its object URL', () => {
  const { controller, revoked } = makeController()
  startStream(controller)
  const firstSource = source(controller)

  controller.handleMessage(openMsg({ session_id: 's2' }))
  assert.deepEqual(revoked, ['blob:fake/1'])
  assert.notEqual(source(controller), firstSource)
  assert.equal(controller.sessionId, 's2')
  assert.equal(controller.sequencer.inGap, false)
  controller.destroy()
})

test('a re-delivered open does not restart the running session', () => {
  const { controller, revoked } = makeController()
  startStream(controller)
  const firstSource = source(controller)

  controller.handleMessage(openMsg())
  assert.equal(source(controller), firstSource, 'the pipeline was kept')
  assert.deepEqual(revoked, [], 'nothing was torn down')
  assert.equal(buffer(controller).appended.length, 2)
  controller.destroy()
})

test('a resync open rebuilds the pipeline and says so', () => {
  const { controller, revoked } = makeController()
  startStream(controller)
  controller.handleMessage(openMsg({ session_id: 's2', resync: true }))
  assert.deepEqual(revoked, ['blob:fake/1'])
  assert.ok(controller.ui.message.textContent.includes('restarted'))
  controller.destroy()
})

test('an interruption with no stream of its own is ignored', () => {
  const { controller } = makeController()
  controller.onInterrupted()
  assert.equal(state(controller), 'waiting')
  controller.destroy()
})

test('a stream that starts mid-flight asks for a replay instead of guessing', async () => {
  const { controller, resumeCalls } = makeController()
  controller.handleMessage(segMsg(12, 0x0c))
  await flush()
  assert.equal(resumeCalls.length, 1)
  assert.equal(resumeCalls[0].lastSeq, -1)
  controller.destroy()
})

/* ------------------------------------------------------------ terminals */

test('complete ends the media source and keeps the clip', () => {
  const { controller, revoked } = makeController()
  startStream(controller)
  const ms = source(controller)
  controller.handleMessage(endMsg(3, 'complete', { segments: 1 }))
  ms.settle(2)
  assert.equal(ms.endedCount, 1)
  assert.equal(state(controller), 'complete')
  assert.deepEqual(revoked, [], 'the finished clip stays playable')
  controller.destroy()
})

test('complete warns when fewer segments arrived than were sent', () => {
  const { controller } = makeController()
  startStream(controller)
  controller.handleMessage(endMsg(3, 'complete', { segments: 9 }))
  assert.ok(controller.ui.message.textContent.includes('1 of 9 segments'))
  controller.destroy()
})

test('cancel stops now: queue dropped, buffer released, URL revoked', () => {
  const { controller, revoked } = makeController()
  startStream(controller)
  const video = controller.ui.video
  controller.handleMessage(segMsg(3, 0x03))
  controller.handleMessage(endMsg(4, 'cancelled'))

  assert.equal(state(controller), 'cancelled')
  assert.deepEqual(revoked, ['blob:fake/1'])
  assert.equal(controller.pipeline.queue.length, 0)
  assert.equal(video.paused, true)
  assert.equal(video.src, '')
  assert.equal(
    video.listenerCount(),
    3,
    'only the widget listeners remain: two frame watchers and the mute sync',
  )
  controller.destroy()
})

test('an interrupted run is treated as a cancellation', () => {
  const { controller, revoked } = makeController()
  startStream(controller)
  controller.onInterrupted()
  assert.equal(state(controller), 'cancelled')
  assert.deepEqual(revoked, ['blob:fake/1'])
  controller.destroy()
})

test('a stream error keeps what decoded and says so once', () => {
  const { controller, revoked } = makeController()
  startStream(controller)
  const ms = source(controller)
  controller.handleMessage(endMsg(3, 'error', { message: 'muxer failed' }))
  ms.settle(2)
  assert.equal(state(controller), 'error')
  assert.equal(controller.ui.message.textContent, 'muxer failed')
  assert.equal(controller.ui.message.dataset.severity, 'error')
  assert.deepEqual(revoked, [])
  assert.equal(ms.endedCount, 1)
  controller.destroy()
})

test('a terminal state survives a dropped socket', () => {
  const { controller } = makeController()
  startStream(controller)
  controller.handleMessage(endMsg(3, 'complete'))
  controller.setConnection('reconnecting')
  assert.equal(state(controller), 'complete')
  controller.destroy()
})

test('a dropped socket during sampling shows reconnecting, then resumes', async () => {
  const { controller, resumeCalls } = makeController()
  startStream(controller)
  controller.handleMessage(statusMsg(3, 'sampling'))
  controller.setConnection('reconnecting')
  assert.equal(state(controller), 'reconnecting')
  controller.setConnection('online')
  await flush()
  assert.equal(resumeCalls.length, 1)
  assert.equal(resumeCalls[0].lastSeq, 3)
  controller.destroy()
})

/* ------------------------------------------------------------- controls */

test('the player starts muted and the unmute control turns sound on', () => {
  const { controller } = makeController()
  const video = controller.ui.video
  assert.equal(video.muted, true)
  assert.equal(controller.ui.muteBtn.getAttribute('aria-pressed'), 'false')
  assert.equal(controller.ui.muteText.textContent, 'Unmute')

  controller.ui.muteBtn.emit('click')
  assert.equal(video.muted, false)
  assert.equal(controller.ui.muteBtn.getAttribute('aria-pressed'), 'true')

  controller.ui.muteBtn.emit('click')
  assert.equal(video.muted, true)
  controller.destroy()
})

test('unmuting a silenced element restores its volume', () => {
  const { controller } = makeController()
  const video = controller.ui.video
  video.volume = 0
  controller.ui.muteBtn.emit('click')
  assert.equal(video.muted, false)
  assert.equal(video.volume, 1)
  controller.destroy()
})

test('a refused autoplay offers an explicit Play control', async () => {
  const { controller } = makeController()
  controller.ui.video.playRejects = true
  startStream(controller)
  await flush()
  assert.equal(controller.ui.action.hidden, false)
  assert.equal(controller.ui.action.textContent, 'Play')
  controller.destroy()
})

test('a manual retry asks for a resume immediately', async () => {
  const { controller, resumeCalls } = makeController()
  startStream(controller)
  controller.ui.setAction('Retry')
  controller.ui.action.emit('click')
  await flush()
  assert.equal(resumeCalls.length, 1)
  controller.destroy()
})

/* ------------------------------------------------------------ lifecycle */

test('a new run clears the previous preview', () => {
  const { controller, revoked } = makeController()
  startStream(controller)
  controller.onExecutionStart()
  assert.equal(state(controller), 'waiting')
  assert.equal(controller.sessionId, null)
  assert.deepEqual(revoked, ['blob:fake/1'])
  assert.equal(controller.ui.root.classList.contains('rvp--has-frames'), false)
  controller.destroy()
})

test('destroy releases the URL, the listeners and the element', () => {
  const { controller, revoked } = makeController()
  startStream(controller)
  const root = controller.ui.root
  const video = controller.ui.video
  controller.destroy()

  assert.deepEqual(revoked, ['blob:fake/1'])
  assert.equal(video.listenerCount(), 0)
  assert.equal(root.totalListeners(), 0)
  assert.equal(root.parent, null)
  assert.equal(controller.node, null)
})

test('destroy is idempotent and messages after it are ignored', () => {
  const { controller } = makeController()
  startStream(controller)
  controller.destroy()
  controller.destroy()
  controller.handleMessage(segMsg(3, 0x03))
  assert.equal(controller.destroyed, true)
})

test('a browser that cannot decode the stream still shows the phase', () => {
  const { controller } = makeController()
  controller.handleMessage(openMsg({ mime: 'video/mp4; codecs="hev1.1.6.L93.B0"' }))
  assert.equal(state(controller), 'error')
  assert.ok(controller.ui.message.textContent.includes('cannot decode'))

  // Media is skipped from here on, but the run's own phases keep updating.
  controller.handleMessage(initMsg(1))
  controller.handleMessage(segMsg(2, 0x01))
  controller.handleMessage(endMsg(3, 'complete'))
  assert.equal(state(controller), 'complete')
  controller.destroy()
})

/* ----------------------------------------------------------------- run */

let failed = 0
for (const [name, fn] of tests) {
  try {
    await fn()
    console.log(`ok   ${name}`)
  } catch (err) {
    failed += 1
    console.log(`FAIL ${name}\n     ${err && err.message}`)
  }
}
dom.uninstall()
console.log(`\n${tests.length - failed}/${tests.length} checks passed`)
process.exit(failed === 0 ? 0 : 1)
