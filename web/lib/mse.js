/**
 * MediaSource pipeline: init segment + ordered fMP4 fragments into a <video>.
 *
 * Browser module (needs MediaSource and a media element), but every external
 * dependency is injected through `env`, so the queueing and teardown logic can
 * be driven by fakes. No ComfyUI imports, no import-time side effects.
 *
 * Contract with the caller: `appendInit` exactly once per source, then
 * `appendSegment` in stream order. Ordering, de-duplication and gap handling
 * happen upstream in `sequencer.js`; this file only guarantees that what it is
 * given is appended one at a time, in the order received, and that a failure
 * ends up as a phase change instead of an exception escaping into ComfyUI.
 */

/** Seconds of already-played media kept when evicting under quota pressure. */
const EVICT_KEEP_BEHIND = 6

/** Drift past which live playback jumps forward to the buffer edge. */
const LIVE_CATCHUP_THRESHOLD = 8

/** Where the catch-up jump lands, measured back from the buffer edge. */
const LIVE_CATCHUP_TARGET = 1.5

/** How many times one fragment may be retried after trimming the buffer. */
const MAX_QUOTA_RETRIES = 3

function defaultEnv() {
  return {
    MediaSource:
      typeof window !== 'undefined'
        ? window.ManagedMediaSource || window.MediaSource
        : undefined,
    isManaged:
      typeof window !== 'undefined' &&
      !!window.ManagedMediaSource &&
      !window.MediaSource,
    createObjectURL: (obj) => URL.createObjectURL(obj),
    revokeObjectURL: (url) => URL.revokeObjectURL(url),
  }
}

export class MediaPipeline {
  /**
   * @param {object} options
   * @param {HTMLVideoElement} options.video
   * @param {(phase: string, info?: object) => void} [options.onPhase]
   * @param {(message: string, err?: unknown) => void} [options.onError]
   * @param {(stats: object) => void} [options.onStats]
   * @param {object} [options.env]
   */
  constructor({ video, onPhase, onError, onStats, env } = {}) {
    this.video = video
    this.onPhase = onPhase || (() => {})
    this.onError = onError || (() => {})
    this.onStats = onStats || (() => {})
    this.env = { ...defaultEnv(), ...(env || {}) }

    this.mediaSource = null
    this.sourceBuffer = null
    this.objectUrl = null
    this.mime = null
    this.queue = []
    this.destroyed = false
    this.ended = false
    this.initAppended = false
    this.appendedBytes = 0
    this.appendedSegments = 0
    /** Set once the viewer takes manual control (pause, seek, volume seek). */
    this.userControlled = false
    this._listeners = []
    this._phase = 'idle'
    this._quotaRetries = 0
  }

  get phase() {
    return this._phase
  }

  static isTypeSupported(mime, env) {
    const resolved = { ...defaultEnv(), ...(env || {}) }
    const MS = resolved.MediaSource
    if (!MS || typeof MS.isTypeSupported !== 'function') return false
    try {
      return MS.isTypeSupported(mime)
    } catch {
      return false
    }
  }

  _setPhase(phase, info) {
    if (this.destroyed || this._phase === phase) return
    this._phase = phase
    try {
      this.onPhase(phase, info)
    } catch {
      /* a UI callback must never break the pipeline */
    }
  }

  _on(target, type, handler, options) {
    if (!target || typeof target.addEventListener !== 'function') return
    target.addEventListener(type, handler, options)
    this._listeners.push([target, type, handler, options])
  }

  _offAll() {
    for (const [target, type, handler, options] of this._listeners) {
      try {
        target.removeEventListener(type, handler, options)
      } catch {
        /* the element may already be gone */
      }
    }
    this._listeners = []
  }

  /**
   * Attach a fresh MediaSource for `mime`.
   * @returns {boolean} false when the browser cannot play this format
   */
  open(mime) {
    if (this.destroyed) return false
    this.close()

    const MS = this.env.MediaSource
    if (!MS) {
      this._fail('This browser has no MediaSource support, so the live preview is unavailable.')
      return false
    }
    if (!MediaPipeline.isTypeSupported(mime, this.env)) {
      this._fail(`This browser cannot decode ${mime}.`)
      return false
    }

    this.mime = mime
    this.ended = false
    this.initAppended = false
    this.queue = []
    this.appendedBytes = 0
    this.appendedSegments = 0

    try {
      this.mediaSource = new MS()
      this.objectUrl = this.env.createObjectURL(this.mediaSource)
      if (this.env.isManaged && this.video) {
        // ManagedMediaSource (Safari) refuses to attach while the element may
        // hand playback to a remote target.
        this.video.disableRemotePlayback = true
      }
      if (this.video) this.video.src = this.objectUrl
      this._on(this.mediaSource, 'sourceopen', () => this._onSourceOpen())
      this._attachVideoListeners()
      this._setPhase('starting')
      return true
    } catch (err) {
      this._fail('Could not start the preview decoder.', err)
      return false
    }
  }

  _onSourceOpen() {
    if (this.destroyed || !this.mediaSource || this.sourceBuffer) return
    try {
      const sb = this.mediaSource.addSourceBuffer(this.mime)
      sb.mode = 'segments'
      this.sourceBuffer = sb
      this._on(sb, 'updateend', () => this._pump())
      this._on(sb, 'error', () => this._fail('The decoder rejected a media fragment.'))
      this._pump()
    } catch (err) {
      this._fail('Could not create a decoder for this stream.', err)
    }
  }

  _attachVideoListeners() {
    const video = this.video
    if (!video) return
    this._on(video, 'playing', () => this._setPhase('playing'))
    this._on(video, 'waiting', () => {
      if (!this.ended) this._setPhase('stalled')
    })
    this._on(video, 'stalled', () => {
      if (!this.ended) this._setPhase('stalled')
    })
    this._on(video, 'seeking', () => {
      this.userControlled = true
    })
    this._on(video, 'pause', () => {
      if (!this.ended) this.userControlled = true
    })
    this._on(video, 'error', () => {
      const err = video.error
      this._fail(err ? `Playback failed (code ${err.code}).` : 'Playback failed.')
    })
  }

  /** Queue the init segment (ftyp + moov). */
  appendInit(bytes) {
    if (this.initAppended) {
      this.onError('A second init segment arrived for one stream; it was ignored.')
      return
    }
    this.initAppended = true
    this._enqueue(bytes)
  }

  /** Queue one media fragment (moof + mdat). */
  appendSegment(bytes) {
    if (!this.initAppended) {
      this.onError('A media fragment arrived before the init segment; it was dropped.')
      return
    }
    this._enqueue(bytes)
    this.appendedSegments += 1
  }

  _enqueue(bytes) {
    if (this.destroyed || this.ended) return
    if (!bytes || bytes.length === 0) return
    this.queue.push(bytes)
    this._pump()
  }

  _pump() {
    if (this.destroyed || !this.sourceBuffer) return
    if (this.sourceBuffer.updating) return

    if (this.queue.length === 0) {
      if (this._pendingEnd) this._finishEndOfStream()
      else this._afterAppend()
      return
    }

    const chunk = this.queue.shift()
    try {
      this.sourceBuffer.appendBuffer(chunk)
      this.appendedBytes += chunk.length
      this._quotaRetries = 0
    } catch (err) {
      if (err && err.name === 'QuotaExceededError') {
        // Bounded: eviction that keeps succeeding while the append keeps
        // failing would otherwise spin forever on `updateend`.
        if (this._quotaRetries < MAX_QUOTA_RETRIES && this._evict()) {
          this._quotaRetries += 1
          // Retry once the eviction settles; `updateend` re-enters `_pump`.
          this.queue.unshift(chunk)
          return
        }
        this._fail('The preview buffer is full and could not be trimmed.', err)
        return
      }
      this._fail('A media fragment could not be appended.', err)
    }
  }

  /**
   * Drop already-played media so a long stream does not hit the browser's
   * per-buffer quota. Returns false when there is nothing safe to remove.
   */
  _evict() {
    const sb = this.sourceBuffer
    const video = this.video
    if (!sb || !video) return false
    try {
      const buffered = sb.buffered
      if (!buffered || buffered.length === 0) return false
      const start = buffered.start(0)
      const cutoff = Math.max(start, (video.currentTime || 0) - EVICT_KEEP_BEHIND)
      if (cutoff - start < 1) return false
      sb.remove(start, cutoff)
      return true
    } catch {
      return false
    }
  }

  _afterAppend() {
    this._catchUpToLiveEdge()
    this._reportStats()
  }

  /**
   * Keep a live stream near its buffer edge. Skipped entirely once the viewer
   * has paused or seeked - their playhead is theirs.
   */
  _catchUpToLiveEdge() {
    const video = this.video
    const sb = this.sourceBuffer
    if (!video || !sb || this.ended || this.userControlled) return
    try {
      const buffered = video.buffered
      if (!buffered || buffered.length === 0) return
      const edge = buffered.end(buffered.length - 1)
      if (edge - (video.currentTime || 0) > LIVE_CATCHUP_THRESHOLD) {
        video.currentTime = Math.max(0, edge - LIVE_CATCHUP_TARGET)
      }
    } catch {
      /* buffered can throw while the source is closing */
    }
  }

  _reportStats() {
    let bufferedSeconds = 0
    try {
      const buffered = this.video && this.video.buffered
      if (buffered && buffered.length > 0) {
        bufferedSeconds = Math.max(
          0,
          buffered.end(buffered.length - 1) - (this.video.currentTime || 0),
        )
      }
    } catch {
      /* ignore */
    }
    try {
      this.onStats({
        segments: this.appendedSegments,
        bytes: this.appendedBytes,
        queued: this.queue.length,
        bufferedSeconds,
      })
    } catch {
      /* a UI callback must never break the pipeline */
    }
  }

  /** Report a gap in the stream while the client waits for the missing seq. */
  markGap() {
    if (!this.ended && !this.destroyed) this._setPhase('gap')
  }

  /** Leave gap state without changing anything else. */
  clearGap() {
    if (this._phase === 'gap') {
      this._setPhase(this.video && !this.video.paused ? 'playing' : 'stalled')
    }
  }

  /** Signal that no further fragments will arrive; drains the queue first. */
  endOfStream() {
    if (this.destroyed || this.ended) return
    this._pendingEnd = true
    this._pump()
  }

  _finishEndOfStream() {
    this._pendingEnd = false
    this.ended = true
    try {
      if (this.mediaSource && this.mediaSource.readyState === 'open') {
        this.mediaSource.endOfStream()
      }
    } catch {
      /* a closing source is not an error worth surfacing */
    }
    this._reportStats()
  }

  _fail(message, err) {
    if (this.destroyed) return
    this._setPhase('failed', { message })
    try {
      this.onError(message, err)
    } catch {
      /* ignore */
    }
  }

  /**
   * Stop immediately and release everything: pending appends, the source
   * buffer, the object URL and every listener. Safe to call repeatedly.
   */
  close() {
    const hadSource = !!(this.mediaSource || this.objectUrl)
    this.queue = []
    this._pendingEnd = false
    this._offAll()

    const sb = this.sourceBuffer
    const ms = this.mediaSource
    this.sourceBuffer = null
    this.mediaSource = null

    if (sb && ms) {
      try {
        if (sb.updating) sb.abort()
      } catch {
        /* ignore */
      }
      try {
        if (ms.readyState === 'open') ms.removeSourceBuffer(sb)
      } catch {
        /* ignore */
      }
    }
    if (ms) {
      try {
        if (ms.readyState === 'open') ms.endOfStream()
      } catch {
        /* ignore */
      }
    }

    // Detaching is only meaningful if something was attached; a pristine
    // element must not be reloaded just because open() cleans up first.
    const video = hadSource ? this.video : null
    if (video) {
      try {
        video.pause()
      } catch {
        /* ignore */
      }
      try {
        video.removeAttribute('src')
        video.load()
      } catch {
        /* ignore */
      }
    }

    if (this.objectUrl) {
      try {
        this.env.revokeObjectURL(this.objectUrl)
      } catch {
        /* ignore */
      }
      this.objectUrl = null
    }

    this.initAppended = false
    this.ended = false
    this.userControlled = false
    this._quotaRetries = 0
    this._phase = 'idle'
  }

  destroy() {
    if (this.destroyed) return
    this.close()
    this.destroyed = true
    this.video = null
  }
}

export const MSE_TUNING = Object.freeze({
  EVICT_KEEP_BEHIND,
  LIVE_CATCHUP_THRESHOLD,
  LIVE_CATCHUP_TARGET,
  MAX_QUOTA_RETRIES,
})
