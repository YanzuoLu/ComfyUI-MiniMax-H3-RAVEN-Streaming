/**
 * DOM for the in-node preview.
 *
 * Browser module (builds elements), but it takes no ComfyUI import and has no
 * import-time side effects. It owns markup, labels and accessibility only;
 * every decision about *what* to show comes from the controller.
 */

import { describeState } from './states.js'

const SVG_NS = 'http://www.w3.org/2000/svg'

function el(tag, className, attrs) {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined) continue
      if (value === true) node.setAttribute(key, '')
      else node.setAttribute(key, String(value))
    }
  }
  return node
}

function svgIcon(paths, { size = 14 } = {}) {
  const svg = document.createElementNS(SVG_NS, 'svg')
  svg.setAttribute('viewBox', '0 0 24 24')
  svg.setAttribute('width', String(size))
  svg.setAttribute('height', String(size))
  svg.setAttribute('fill', 'none')
  svg.setAttribute('stroke', 'currentColor')
  svg.setAttribute('stroke-width', '2')
  svg.setAttribute('stroke-linecap', 'round')
  svg.setAttribute('stroke-linejoin', 'round')
  svg.setAttribute('aria-hidden', 'true')
  svg.setAttribute('focusable', 'false')
  for (const d of paths) {
    const path = document.createElementNS(SVG_NS, 'path')
    path.setAttribute('d', d)
    svg.appendChild(path)
  }
  return svg
}

const ICON_SPEAKER_MUTED = [
  'M11 5 6 9H3v6h3l5 4z',
  'M17 9l4 6',
  'M21 9l-4 6',
]

const ICON_SPEAKER_ON = [
  'M11 5 6 9H3v6h3l5 4z',
  'M16 8.5a4.5 4.5 0 0 1 0 7',
  'M19 6a8 8 0 0 1 0 12',
]

function formatBytes(bytes) {
  if (!bytes) return '0 B'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export class PreviewUI {
  /**
   * @param {object} handlers
   * @param {() => void} handlers.onToggleMute
   * @param {() => void} handlers.onAction  retry / resume
   */
  constructor({ onToggleMute, onAction } = {}) {
    this.onToggleMute = onToggleMute || (() => {})
    this.onAction = onAction || (() => {})
    this._listeners = []
    this._build()
  }

  _build() {
    const root = el('div', 'rvp', {
      role: 'group',
      'aria-label': 'RAVEN streaming preview',
    })

    const stage = el('div', 'rvp__stage')
    const video = el('video', 'rvp__video', {
      playsinline: true,
      controls: true,
      preload: 'none',
      tabindex: '0',
      'aria-label': 'Streamed preview video',
    })
    video.muted = true
    video.autoplay = true
    video.disablePictureInPicture = false

    const overlay = el('div', 'rvp__overlay')
    const spinner = el('span', 'rvp__spinner', { 'aria-hidden': 'true' })
    const overlayText = el('span', 'rvp__overlay-text')
    overlayText.textContent = 'Waiting for the first frame'
    overlay.append(spinner, overlayText)

    stage.append(video, overlay)

    const bar = el('div', 'rvp__bar')

    const status = el('span', 'rvp__status', {
      role: 'status',
      'aria-live': 'polite',
    })
    const dot = el('span', 'rvp__dot', { 'aria-hidden': 'true' })
    const label = el('span', 'rvp__label')
    label.textContent = 'Waiting'
    status.append(dot, label)

    const meta = el('span', 'rvp__meta', { title: 'Segments appended / buffer ahead' })

    const muteBtn = el('button', 'rvp__mute is-muted', {
      type: 'button',
      'aria-pressed': 'false',
      title: 'Sound is off. Click to unmute (the video keeps its own volume and playback controls).',
    })
    const muteIcon = el('span', 'rvp__mute-icon')
    muteIcon.appendChild(svgIcon(ICON_SPEAKER_MUTED))
    const muteText = el('span', 'rvp__mute-text')
    muteText.textContent = 'Unmute'
    muteBtn.append(muteIcon, muteText)

    const action = el('button', 'rvp__action', { type: 'button', hidden: true })
    action.textContent = 'Retry'

    bar.append(status, meta, muteBtn, action)

    const message = el('p', 'rvp__message', { hidden: true })

    root.append(stage, bar, message)

    this.root = root
    this.stage = stage
    this.video = video
    this.overlay = overlay
    this.overlayText = overlayText
    this.statusEl = status
    this.dot = dot
    this.label = label
    this.meta = meta
    this.muteBtn = muteBtn
    this.muteIcon = muteIcon
    this.muteText = muteText
    this.action = action
    this.message = message

    this._on(muteBtn, 'click', (event) => {
      event.preventDefault()
      event.stopPropagation()
      this.onToggleMute()
    })
    this._on(action, 'click', (event) => {
      event.preventDefault()
      event.stopPropagation()
      this.onAction()
    })
    this._on(video, 'volumechange', () => this.syncMuteButton())

    // The canvas swallows pointer and key events for panning and shortcuts;
    // stopping propagation here is what keeps the native video controls and
    // the buttons usable inside a node.
    for (const type of ['pointerdown', 'pointerup', 'click', 'dblclick', 'wheel']) {
      this._on(stage, type, (event) => event.stopPropagation())
    }
    this._on(root, 'keydown', (event) => event.stopPropagation())

    this._resizeObserver = null
    if (typeof ResizeObserver !== 'undefined') {
      this._resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const width = entry.contentRect ? entry.contentRect.width : 0
          root.classList.toggle('rvp--narrow', width > 0 && width < 230)
          root.classList.toggle('rvp--tight', width > 0 && width < 160)
        }
      })
      this._resizeObserver.observe(root)
    }
  }

  _on(target, type, handler, options) {
    target.addEventListener(type, handler, options)
    this._listeners.push([target, type, handler, options])
  }

  /** Aspect ratio drives the node's preferred height. */
  setAspect(width, height) {
    if (!width || !height) return
    this.root.style.setProperty('--rvp-aspect', `${width} / ${height}`)
  }

  setState(state, detail) {
    const info = describeState(state)
    this.root.dataset.state = state
    this.root.dataset.tone = info.tone
    this.label.textContent = info.label
    this.statusEl.setAttribute(
      'aria-label',
      detail ? `${info.label}. ${detail}` : info.label,
    )
    const text = detail || info.detail || info.label
    this.overlayText.textContent = text
    this.root.classList.toggle('rvp--busy', info.tone === 'busy')
  }

  /** Hide the placeholder once the decoder has produced a frame. */
  setHasFrames(hasFrames) {
    this.root.classList.toggle('rvp--has-frames', !!hasFrames)
  }

  setStats({ segments = 0, bufferedSeconds = 0, bytes = 0, dropped = 0 } = {}) {
    const parts = [`${segments} seg`]
    if (bufferedSeconds > 0) parts.push(`${bufferedSeconds.toFixed(1)}s buffered`)
    if (bytes > 0) parts.push(formatBytes(bytes))
    if (dropped > 0) parts.push(`${dropped} dropped`)
    this.meta.textContent = parts.join(' · ')
    this.meta.title = `${segments} segments appended, ${formatBytes(bytes)} decoded, ${bufferedSeconds.toFixed(1)}s ahead of the playhead${dropped ? `, ${dropped} duplicate or late packets dropped` : ''}`
  }

  /**
   * @param {string|null} text
   * @param {'info'|'warn'|'error'} [severity]
   */
  setMessage(text, severity = 'info') {
    if (!text) {
      this.message.hidden = true
      this.message.textContent = ''
      this.message.removeAttribute('role')
      return
    }
    this.message.hidden = false
    this.message.dataset.severity = severity
    this.message.setAttribute('role', severity === 'error' ? 'alert' : 'status')
    this.message.textContent = text
  }

  /** @param {string|null} label  null hides the button */
  setAction(label) {
    if (!label) {
      this.action.hidden = true
      return
    }
    this.action.hidden = false
    this.action.textContent = label
    this.action.setAttribute('aria-label', label)
  }

  /** Reflect the element's real muted state; the element stays the source of truth. */
  syncMuteButton() {
    const muted = !this.video || this.video.muted || this.video.volume === 0
    this.muteBtn.classList.toggle('is-muted', muted)
    this.muteBtn.setAttribute('aria-pressed', muted ? 'false' : 'true')
    this.muteText.textContent = muted ? 'Unmute' : 'Sound on'
    this.muteBtn.title = muted
      ? 'Sound is off. Click to unmute (the video keeps its own volume and playback controls).'
      : 'Sound is on. Click to mute.'
    this.muteIcon.replaceChildren(svgIcon(muted ? ICON_SPEAKER_MUTED : ICON_SPEAKER_ON))
  }

  /** Height the node needs for the current width, used by the DOM widget. */
  preferredHeight(width) {
    const aspect = this.root.style.getPropertyValue('--rvp-aspect') || '16 / 9'
    const [w, h] = aspect.split('/').map((part) => parseFloat(part.trim()))
    const ratio = w > 0 && h > 0 ? h / w : 9 / 16
    const media = Math.round(Math.max(80, (width || 240) * ratio))
    const chrome = this.message.hidden ? 30 : 52
    return media + chrome
  }

  destroy() {
    for (const [target, type, handler, options] of this._listeners) {
      try {
        target.removeEventListener(type, handler, options)
      } catch {
        /* ignore */
      }
    }
    this._listeners = []
    if (this._resizeObserver) {
      try {
        this._resizeObserver.disconnect()
      } catch {
        /* ignore */
      }
      this._resizeObserver = null
    }
    try {
      this.root.remove()
    } catch {
      /* ignore */
    }
  }
}
