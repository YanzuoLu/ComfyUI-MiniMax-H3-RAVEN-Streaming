/**
 * Test doubles shared by the Node harnesses.
 *
 * Two groups:
 *   - media fakes (MediaSource / SourceBuffer), enough to drive the append
 *     queue deterministically: appends settle only when a test says so.
 *   - a small DOM, enough for `lib/ui.js` to build its markup and for the
 *     controller to drive it. It is not a browser and does not pretend to be
 *     one: no layout, no CSS, no real playback.
 */

/* ---------------------------------------------------------------- events */

export class FakeEventTarget {
  constructor() {
    this._handlers = new Map()
  }
  addEventListener(type, handler) {
    if (!this._handlers.has(type)) this._handlers.set(type, new Set())
    this._handlers.get(type).add(handler)
  }
  removeEventListener(type, handler) {
    const set = this._handlers.get(type)
    if (set) set.delete(handler)
  }
  emit(type, extra = {}) {
    for (const handler of [...(this._handlers.get(type) || [])]) {
      handler({ type, preventDefault() {}, stopPropagation() {}, ...extra })
    }
  }
  dispatchEvent(event) {
    this.emit(event.type, event)
    return true
  }
  listenerCount() {
    let total = 0
    for (const set of this._handlers.values()) total += set.size
    return total
  }
}

/* ----------------------------------------------------------------- media */

export class FakeSourceBuffer extends FakeEventTarget {
  constructor(mime, source) {
    super()
    this.mime = mime
    this.source = source
    this.updating = false
    this.appended = []
    this.removed = []
    this.aborted = false
    /** Throw QuotaExceededError until this many appends have succeeded. */
    this.quotaUntil = 0
    this.buffered = { length: 1, start: () => 0, end: () => 30 }
  }
  appendBuffer(bytes) {
    if (this.appended.length < this.quotaUntil) {
      const err = new Error('quota')
      err.name = 'QuotaExceededError'
      throw err
    }
    this.appended.push(bytes)
    this.updating = true
    this.source._pendingUpdates.push(() => {
      this.updating = false
      this.emit('updateend')
    })
  }
  remove(start, end) {
    this.removed.push([start, end])
    // Trimming frees room, which is what makes the retry succeed.
    this.quotaUntil = 0
    this.updating = true
    this.source._pendingUpdates.push(() => {
      this.updating = false
      this.emit('updateend')
    })
  }
  abort() {
    this.aborted = true
    this.updating = false
  }
}

export class FakeMediaSource extends FakeEventTarget {
  constructor() {
    super()
    this.readyState = 'closed'
    this.sourceBuffers = []
    this.endedCount = 0
    this._pendingUpdates = []
  }
  static isTypeSupported(mime) {
    return typeof mime === 'string' && mime.includes('avc1')
  }
  addSourceBuffer(mime) {
    const sb = new FakeSourceBuffer(mime, this)
    this.sourceBuffers.push(sb)
    return sb
  }
  removeSourceBuffer(sb) {
    this.sourceBuffers = this.sourceBuffers.filter((item) => item !== sb)
  }
  endOfStream() {
    this.endedCount += 1
    this.readyState = 'ended'
  }
  /** Move to the state where a source buffer can be created. */
  openNow() {
    this.readyState = 'open'
    this.emit('sourceopen')
  }
  /** Complete every append/remove that is in flight. */
  settle(rounds = 1) {
    for (let i = 0; i < rounds; i++) {
      while (this._pendingUpdates.length) this._pendingUpdates.shift()()
    }
  }
}

/** A MediaPipeline `env` backed by FakeMediaSource, with URL bookkeeping. */
export function makeMediaEnv() {
  const created = []
  const revoked = []
  return {
    created,
    revoked,
    env: {
      MediaSource: FakeMediaSource,
      isManaged: false,
      createObjectURL: (obj) => {
        created.push(obj)
        return `blob:fake/${created.length}`
      },
      revokeObjectURL: (url) => revoked.push(url),
    },
  }
}

/* ------------------------------------------------------------------- DOM */

class FakeClassList {
  constructor(el) {
    this.el = el
    this.set = new Set()
  }
  add(...names) {
    for (const name of names) if (name) this.set.add(name)
    this._sync()
  }
  remove(...names) {
    for (const name of names) this.set.delete(name)
    this._sync()
  }
  toggle(name, force) {
    const on = force === undefined ? !this.set.has(name) : !!force
    if (on) this.set.add(name)
    else this.set.delete(name)
    this._sync()
    return on
  }
  contains(name) {
    return this.set.has(name)
  }
  _sync() {
    this.el._className = [...this.set].join(' ')
  }
}

export class FakeElement extends FakeEventTarget {
  constructor(tagName) {
    super()
    this.tagName = String(tagName).toUpperCase()
    this.children = []
    this.parent = null
    this.attributes = new Map()
    this.dataset = {}
    this.classList = new FakeClassList(this)
    this._className = ''
    this._textContent = ''
    this.hidden = false
    this.style = {
      _props: new Map(),
      setProperty(name, value) {
        this._props.set(name, value)
      },
      getPropertyValue(name) {
        return this._props.get(name) || ''
      },
    }
  }
  get className() {
    return this._className
  }
  set className(value) {
    this._className = value || ''
    this.classList.set = new Set(this._className.split(/\s+/).filter(Boolean))
  }
  get textContent() {
    if (this.children.length === 0) return this._textContent
    return this.children.map((child) => child.textContent).join('')
  }
  set textContent(value) {
    this.children = []
    this._textContent = value == null ? '' : String(value)
  }
  setAttribute(name, value) {
    this.attributes.set(name, String(value))
  }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null
  }
  removeAttribute(name) {
    this.attributes.delete(name)
  }
  appendChild(child) {
    child.parent = this
    this.children.push(child)
    return child
  }
  append(...nodes) {
    for (const node of nodes) this.appendChild(node)
  }
  replaceChildren(...nodes) {
    this.children = []
    for (const node of nodes) this.appendChild(node)
  }
  remove() {
    if (!this.parent) return
    this.parent.children = this.parent.children.filter((child) => child !== this)
    this.parent = null
  }
  querySelector() {
    return null
  }
  /** Every listener attached anywhere in this subtree. */
  totalListeners() {
    return this.children.reduce(
      (total, child) => total + child.totalListeners(),
      this.listenerCount(),
    )
  }
}

export class FakeVideoElement extends FakeElement {
  constructor() {
    super('video')
    this.currentTime = 0
    this.paused = true
    this.muted = false
    this.volume = 1
    this.src = ''
    this.loadCount = 0
    this.playCount = 0
    this.playRejects = false
    this.error = null
    this.buffered = { length: 1, start: () => 0, end: () => 30 }
  }
  play() {
    this.playCount += 1
    if (this.playRejects) return Promise.reject(new Error('autoplay blocked'))
    this.paused = false
    return Promise.resolve()
  }
  pause() {
    this.paused = true
  }
  load() {
    this.loadCount += 1
  }
  removeAttribute(name) {
    super.removeAttribute(name)
    if (name === 'src') this.src = ''
  }
}

/**
 * Install a fake `document` (and a no-op ResizeObserver hole) on globalThis.
 * Returns a handle with the created root elements and an uninstall function.
 */
export function installFakeDom() {
  const previousDocument = globalThis.document
  const created = []
  const head = new FakeElement('head')

  const document = {
    head,
    createElement(tag) {
      const el = tag === 'video' ? new FakeVideoElement() : new FakeElement(tag)
      created.push(el)
      return el
    },
    createElementNS(_ns, tag) {
      const el = new FakeElement(tag)
      created.push(el)
      return el
    },
    querySelector() {
      return null
    },
  }

  globalThis.document = document
  return {
    document,
    created,
    uninstall() {
      globalThis.document = previousDocument
    },
  }
}

/**
 * Replace global timers with ones the test drives by hand, so a 4 s gap
 * timeout costs no wall-clock time.
 */
export function installFakeTimers() {
  const realSetTimeout = globalThis.setTimeout
  const realClearTimeout = globalThis.clearTimeout
  const scheduled = new Map()
  let nextId = 1

  globalThis.setTimeout = (fn, delay) => {
    const id = nextId++
    scheduled.set(id, { fn, delay })
    return id
  }
  globalThis.clearTimeout = (id) => {
    scheduled.delete(id)
  }

  return {
    get pending() {
      return scheduled.size
    },
    /** Fire every timer that is due at or before `ms`. */
    advance(ms) {
      for (const [id, entry] of [...scheduled.entries()]) {
        if (entry.delay <= ms) {
          scheduled.delete(id)
          entry.fn()
        }
      }
    },
    uninstall() {
      globalThis.setTimeout = realSetTimeout
      globalThis.clearTimeout = realClearTimeout
    },
  }
}
