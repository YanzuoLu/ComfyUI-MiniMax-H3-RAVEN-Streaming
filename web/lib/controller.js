/**
 * One preview session for one node.
 *
 * Owns the sequencer, the MediaSource pipeline and the widget UI, and is the
 * only place where the three meet. Everything that can throw is contained
 * here: a preview failure must never propagate into ComfyUI's event dispatch,
 * and it must never affect the node's real outputs.
 *
 * No ComfyUI import - the entry point injects what it needs (`requestResume`,
 * the node handle). That keeps this file drivable from a test harness.
 */

import { MediaPipeline } from './mse.js'
import { PreviewUI } from './ui.js'
import { Sequencer, PUSH_DUPLICATE, PUSH_HELD, PUSH_OVERFLOW } from './sequencer.js'
import { parseEnvelope, ProtocolError } from './protocol.js'
import { resolveState, isTerminal } from './states.js'

/** How long a hole in the sequence is tolerated before asking for a resume. */
const GAP_TIMEOUT_MS = 4000

/** Wait before a second automatic resume request for the same hole. */
const RESUME_RETRY_MS = 8000

export class PreviewController {
  /**
   * @param {object} options
   * @param {object} options.node                 the LGraphNode (read-only use)
   * @param {(req: object) => Promise<object>} [options.requestResume]
   * @param {object} [options.env]                MediaPipeline environment
   * @param {(msg: string, err?: unknown) => void} [options.log]
   */
  constructor({ node, requestResume, env, log } = {}) {
    this.node = node
    this.requestResume = requestResume || null
    this.log = log || (() => {})

    this.sessionId = null
    this.promptId = null
    this.backendPhase = 'waiting'
    this.connection = 'online'
    this.ended = null
    this.resumeUnsupported = false
    /** True when this browser cannot decode the announced stream at all. */
    this.mediaUnavailable = false
    this.destroyed = false
    this.droppedCount = 0
    this.lastStats = { segments: 0, bytes: 0, bufferedSeconds: 0 }
    this._gapTimer = null
    this._lastResumeAt = 0
    this._gapMessageShown = false
    this._videoListeners = []

    this.sequencer = new Sequencer()
    this.ui = new PreviewUI({
      onToggleMute: () => this._toggleMute(),
      onAction: () => this._onAction(),
    })
    this.pipeline = new MediaPipeline({
      video: this.ui.video,
      env,
      onPhase: (phase) => this._onMediaPhase(phase),
      onError: (message, err) => this._onPipelineError(message, err),
      onStats: (stats) => this._onStats(stats),
    })

    this._watchVideo()
    this.ui.syncMuteButton()
    this._render()
  }

  get element() {
    return this.ui.root
  }

  get nodeId() {
    return this.node ? this.node.id : null
  }

  _watchVideo() {
    const video = this.ui.video
    const mark = () => this.ui.setHasFrames(true)
    const add = (type, handler) => {
      video.addEventListener(type, handler)
      this._videoListeners.push([type, handler])
    }
    add('loadeddata', mark)
    add('playing', mark)
  }

  _unwatchVideo() {
    const video = this.ui.video
    if (!video) return
    for (const [type, handler] of this._videoListeners) {
      try {
        video.removeEventListener(type, handler)
      } catch {
        /* ignore */
      }
    }
    this._videoListeners = []
  }

  /* ------------------------------------------------------------- inbound */

  /**
   * Handle one raw websocket message body addressed to this node.
   * Never throws.
   */
  handleMessage(raw) {
    if (this.destroyed) return
    let envelope
    try {
      envelope = parseEnvelope(raw)
    } catch (err) {
      this.droppedCount += 1
      const detail = err instanceof ProtocolError ? err.message : String(err)
      this.log(`dropped a malformed preview message: ${detail}`)
      this._render()
      return
    }

    try {
      this._route(envelope)
    } catch (err) {
      this._onPipelineError('The preview could not process a stream message.', err)
    }
  }

  _route(envelope) {
    if (envelope.event === 'open') {
      // A re-delivered `open` for the session already running must not restart
      // it; only a genuinely new session (or a resync) rebuilds the pipeline.
      const isReplay =
        envelope.sessionId === this.sessionId &&
        envelope.seq <= this.sequencer.lastReleasedSeq
      if (isReplay) {
        this._render()
        return
      }
      this._startSession(envelope)
      return
    }
    if (this.sessionId === null) {
      // Media for a session we never saw the start of: the tab attached late.
      // Ask for a replay rather than appending fragments blind.
      this.sessionId = envelope.sessionId
      this.sequencer.reset(0)
      this._requestResume('joined the stream after it started')
    }
    if (envelope.sessionId !== this.sessionId) {
      this.droppedCount += 1
      this._render()
      return
    }

    const result = this.sequencer.push(envelope.seq, envelope)
    if (result.status === PUSH_DUPLICATE) {
      // Counted by the sequencer itself; counting again here would double it.
      this._render()
      return
    }
    if (result.status === PUSH_OVERFLOW) {
      this.droppedCount += 1
      this.ui.setMessage(
        `Out-of-order backlog is full while segment ${result.missing} is missing.`,
        'warn',
      )
      this._requestResume('the reorder buffer overflowed')
      return
    }
    if (result.status === PUSH_HELD) {
      this._enterGap(result.missing)
      return
    }

    this._clearGapTimer()
    // Clear the gap notice before applying, so a message an applied event sets
    // (an end reason, a backend line) is not wiped straight afterwards.
    if (this._gapMessageShown) {
      this.ui.setMessage(null)
      this._gapMessageShown = false
    }
    for (const { item } of result.released) this._apply(item)
    if (this.sequencer.inGap) this._enterGap(this.sequencer.missingSeq)
    this._render()
  }

  _startSession(envelope) {
    const body = envelope.body
    const isSameSession = envelope.sessionId === this.sessionId
    this._teardownMedia()

    this.sessionId = envelope.sessionId
    this.promptId = envelope.promptId
    this.ended = null
    this.mediaUnavailable = false
    this.droppedCount = isSameSession ? this.droppedCount : 0
    this.sequencer.reset(envelope.seq)
    this.lastStats = { segments: 0, bytes: 0, bufferedSeconds: 0 }
    this.ui.setHasFrames(false)
    this.ui.setAction(null)
    this._gapMessageShown = false
    this.ui.setMessage(
      body.resync ? 'The backend restarted the preview stream.' : null,
      'info',
    )
    if (body.width && body.height) this.ui.setAspect(body.width, body.height)

    // `open` failing (no MediaSource, unsupported codec string) is not fatal to
    // the session: status and end events still drive the label, only the media
    // is skipped. `pipeline.open` has already reported the reason.
    this.mediaUnavailable = !this.pipeline.open(body.mime)

    // The `open` event occupies its own seq; consume it so the following media
    // events line up instead of reading as a permanent gap.
    this.sequencer.push(envelope.seq, envelope)
    this._render()
  }

  _apply(envelope) {
    switch (envelope.event) {
      case 'open':
        // Applied at receive time in _startSession.
        break
      case 'init':
        if (this.mediaUnavailable) break
        this.pipeline.appendInit(envelope.body.bytes)
        this._tryPlay()
        break
      case 'segment':
        if (this.mediaUnavailable) break
        this.pipeline.appendSegment(envelope.body.bytes)
        this._tryPlay()
        break
      case 'status':
        this.backendPhase = envelope.body.phase
        if (envelope.body.message) this.ui.setMessage(envelope.body.message, 'info')
        break
      case 'end':
        this._applyEnd(envelope.body)
        break
      default:
        break
    }
  }

  _applyEnd(body) {
    this.ended = body.reason
    this._clearGapTimer()
    if (body.reason === 'cancelled') {
      // Stop now: drop everything queued, release the buffer and the URL.
      this._teardownMedia()
      this.ui.setHasFrames(false)
      this.ui.setMessage(body.message || 'Sampling was cancelled.', 'info')
    } else if (body.reason === 'error') {
      // Keep what already decoded; the sampler's own error is reported by
      // ComfyUI separately, this line is only about the preview stream.
      this.pipeline.endOfStream()
      this.ui.setMessage(body.message || 'The preview stream ended with an error.', 'error')
    } else {
      this.pipeline.endOfStream()
      const expected = body.segments
      if (expected !== null && expected !== this.pipeline.appendedSegments) {
        this.ui.setMessage(
          `Stream finished with ${this.pipeline.appendedSegments} of ${expected} segments; the preview may be short.`,
          'warn',
        )
      } else {
        this.ui.setMessage(null)
      }
    }
    this.ui.setAction(null)
  }

  /* --------------------------------------------------------------- gaps */

  _enterGap(missingSeq) {
    this.pipeline.markGap()
    this._gapMessageShown = true
    this.ui.setMessage(`Waiting for segment ${missingSeq}.`, 'warn')
    this._render()
    if (this._gapTimer !== null) return
    this._gapTimer = setTimeout(() => {
      this._gapTimer = null
      if (this.destroyed || !this.sequencer.inGap) return
      this._requestResume(`segment ${this.sequencer.missingSeq} did not arrive`)
    }, GAP_TIMEOUT_MS)
  }

  _clearGapTimer() {
    if (this._gapTimer !== null) {
      clearTimeout(this._gapTimer)
      this._gapTimer = null
    }
    this.pipeline.clearGap()
  }

  _requestResume(reason) {
    const now = Date.now()
    if (now - this._lastResumeAt < RESUME_RETRY_MS) return
    this._lastResumeAt = now

    if (!this.requestResume || this.resumeUnsupported) {
      this.ui.setMessage(
        `Preview stream is incomplete (${reason}). Waiting for the backend to resend.`,
        'warn',
      )
      this.ui.setAction('Retry')
      return
    }

    const request = {
      sessionId: this.sessionId,
      nodeId: this.nodeId === null ? null : String(this.nodeId),
      lastSeq: this.sequencer.resumeFrom,
      reason,
    }
    Promise.resolve()
      .then(() => this.requestResume(request))
      .then((result) => {
        if (this.destroyed) return
        if (result && result.supported === false) {
          this.resumeUnsupported = true
          this.ui.setMessage(
            `Preview stream is incomplete (${reason}). This backend does not support resume; waiting for it to resend.`,
            'warn',
          )
          this.ui.setAction('Retry')
          return
        }
        this.ui.setMessage(`Asked the backend to resend from ${request.lastSeq + 1}.`, 'warn')
      })
      .catch((err) => {
        if (this.destroyed) return
        this.log('resume request failed', err)
        this.ui.setMessage(
          `Preview stream is incomplete (${reason}) and the resume request failed.`,
          'warn',
        )
        this.ui.setAction('Retry')
      })
  }

  /* ----------------------------------------------------------- lifecycle */

  /** @param {'online'|'reconnecting'} state */
  setConnection(state) {
    if (this.connection === state) return
    this.connection = state
    if (state === 'online' && this.sessionId && !isTerminal(this._state())) {
      this._requestResume('the connection dropped')
    }
    this._render()
  }

  /** Fired when a new prompt starts: nothing of the old run is valid. */
  onExecutionStart() {
    this._teardownMedia()
    this.sessionId = null
    this.backendPhase = 'waiting'
    this.ended = null
    this.droppedCount = 0
    this.resumeUnsupported = false
    this.mediaUnavailable = false
    this.ui.setHasFrames(false)
    this._gapMessageShown = false
    this.ui.setMessage(null)
    this.ui.setAction(null)
    this.ui.setStats({})
    this._render()
  }

  /** ComfyUI reported the whole run as interrupted. */
  onInterrupted() {
    // A node that never streamed anything has nothing to cancel; saying
    // "Cancelled" there would be noise on every unrelated interruption.
    if (!this.sessionId) return
    if (isTerminal(this._state())) return
    this._applyEnd({ reason: 'cancelled', message: 'Run was cancelled.', segments: null })
    this._render()
  }

  /** ComfyUI reported an execution error for this node. */
  onExecutionError(message) {
    if (this.ended === 'complete') return
    this._applyEnd({
      reason: 'error',
      message: message || 'The run failed; the preview stopped.',
      segments: null,
    })
    this._render()
  }

  /** The node finished executing without the backend closing the stream. */
  onExecuted() {
    if (this.ended || !this.sessionId) return
    this.backendPhase = 'finalizing'
    this._render()
  }

  _teardownMedia() {
    this._clearGapTimer()
    this.sequencer.reset(0)
    try {
      this.pipeline.close()
    } catch (err) {
      this.log('pipeline close failed', err)
    }
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this._clearGapTimer()
    this._unwatchVideo()
    try {
      this.pipeline.destroy()
    } catch (err) {
      this.log('pipeline destroy failed', err)
    }
    try {
      this.ui.destroy()
    } catch (err) {
      this.log('ui destroy failed', err)
    }
    this.node = null
  }

  /* ------------------------------------------------------------ internal */

  _tryPlay() {
    const video = this.ui.video
    if (!video || this.pipeline.userControlled) return
    if (!video.paused) return
    const attempt = video.play()
    if (attempt && typeof attempt.catch === 'function') {
      attempt.catch(() => {
        // Autoplay was refused even while muted; offer an explicit control
        // instead of leaving a still frame with no explanation.
        this.ui.setAction('Play')
      })
    }
  }

  _toggleMute() {
    const video = this.ui.video
    if (!video) return
    const wasMuted = video.muted || video.volume === 0
    video.muted = !wasMuted
    if (!video.muted && video.volume === 0) video.volume = 1
    if (!video.muted) this._tryPlay()
    this.ui.syncMuteButton()
  }

  _onAction() {
    const label = this.ui.action.textContent
    if (label === 'Play') {
      this.ui.setAction(null)
      this.pipeline.userControlled = false
      this._tryPlay()
      return
    }
    this._lastResumeAt = 0
    this._requestResume('a manual retry')
  }

  _onMediaPhase() {
    this._render()
  }

  _onStats(stats) {
    this.lastStats = stats
    this._render()
  }

  _onPipelineError(message, err) {
    if (err) this.log(message, err)
    this.ui.setMessage(message, 'error')
    this._render()
  }

  _state() {
    return resolveState({
      backend: this.backendPhase,
      media: this.pipeline ? this.pipeline.phase : 'idle',
      connection: this.connection,
      ended: this.ended,
    })
  }

  _render() {
    if (this.destroyed) return
    const state = this._state()
    this.ui.setState(state)
    this.ui.setStats({
      segments: this.lastStats.segments || 0,
      bytes: this.lastStats.bytes || 0,
      bufferedSeconds: this.lastStats.bufferedSeconds || 0,
      dropped: this.droppedCount + this.sequencer.stats.duplicates,
    })
  }
}

export const CONTROLLER_TUNING = Object.freeze({ GAP_TIMEOUT_MS, RESUME_RETRY_MS })
