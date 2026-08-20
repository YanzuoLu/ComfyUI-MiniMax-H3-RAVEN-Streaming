/**
 * RAVEN Streaming Sampler - in-node live preview.
 *
 * Entry point: this is the only file that talks to ComfyUI. Everything under
 * ./lib is dependency-free so it can be tested without a browser and without a
 * running server.
 *
 * Verified against ComfyUI c67885b (frontend package 1.49.6):
 *
 *  - `/extensions` serves every *.js under a pack's web directory, so this file
 *    is loaded as an ES module and the sibling ./lib modules resolve relatively
 *    (server.py:359-366, nodes.py:2280-2289).
 *  - `scripts/app.js` / `scripts/api.js` are shipped shims re-exporting the app
 *    and api singletons; both are explicitly exempt from the frontend's
 *    "internal module" deprecation warning (build/plugins/comfyAPIPlugin.ts).
 *  - A JSON websocket message with an unknown `type` is dispatched as a
 *    CustomEvent only if a listener for that exact type is already registered
 *    (api.ts, `default:` branch of the message handler). Listeners are
 *    therefore registered in `setup`, once, before any message can arrive.
 *  - Listener exceptions are caught and logged by the frontend's own wrapper,
 *    so a preview bug cannot break ComfyUI's dispatch. We still guard
 *    everything here, because a caught exception is still a dead preview.
 */

import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'

import { PreviewController } from './lib/controller.js'
import { isSamplerNode, routeToController } from './lib/identity.js'
import { MESSAGE_TYPE, RESUME_ROUTE } from './lib/protocol.js'

const EXTENSION_NAME = 'raven.streaming.preview'
const WIDGET_NAME = 'raven_preview'
const LOG_PREFIX = '[RAVEN preview]'

/** node instance -> controller. A WeakMap would hide the iteration we need. */
const controllers = new Map()

function log(message, err) {
  if (err) console.warn(`${LOG_PREFIX} ${message}`, err)
  else console.warn(`${LOG_PREFIX} ${message}`)
}

function injectStylesheet() {
  const href = new URL('./preview.css', import.meta.url).href
  if (document.querySelector(`link[data-raven-preview="1"]`)) return
  const link = document.createElement('link')
  link.rel = 'stylesheet'
  link.href = href
  link.dataset.ravenPreview = '1'
  document.head.appendChild(link)
}

/**
 * Ask the backend to resend from a known-good seq.
 *
 * The route is optional (see PROTOCOL.md). A 404/405/501 answer means this
 * backend has no resume endpoint, which is reported back to the controller so
 * it can say "waiting for the backend to resend" instead of retrying forever.
 */
async function requestResume(request) {
  let response
  try {
    response = await api.fetchApi(RESUME_ROUTE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: request.sessionId,
        node_id: request.nodeId,
        last_seq: request.lastSeq,
        client_id: api.clientId || null,
        reason: request.reason,
      }),
    })
  } catch (err) {
    throw new Error(`resume request failed: ${err && err.message}`)
  }
  if (response.status === 404 || response.status === 405 || response.status === 501) {
    return { supported: false }
  }
  if (!response.ok) {
    throw new Error(`resume request rejected with ${response.status}`)
  }
  return { supported: true }
}

function attach(node) {
  if (controllers.has(node)) return controllers.get(node)

  const controller = new PreviewController({ node, requestResume, log })
  controllers.set(node, controller)

  try {
    const widget = node.addDOMWidget(WIDGET_NAME, WIDGET_NAME, controller.element, {
      hideOnZoom: true,
      // Not a user input: keep it out of both the workflow JSON and the prompt.
      serialize: false,
      getValue: () => '',
      setValue: () => {},
      getMinHeight: () => {
        const width = Array.isArray(node.size) ? node.size[0] - 20 : 240
        return controller.ui.preferredHeight(width)
      },
    })
    widget.serialize = false
  } catch (err) {
    // A frontend without addDOMWidget leaves the node perfectly usable; only
    // the preview is lost, and it must not leave a detached controller behind.
    log('could not add the preview widget to the node', err)
    detach(node)
    return null
  }

  return controller
}

function detach(node) {
  const controller = controllers.get(node)
  if (!controller) return
  controllers.delete(node)
  try {
    controller.destroy()
  } catch (err) {
    log('failed to tear down a preview', err)
  }
}

function detachAll() {
  for (const node of [...controllers.keys()]) detach(node)
}

/**
 * Drop controllers whose node is no longer on a graph. Loading a workflow
 * rebuilds every node object, and litegraph clears `node.graph` on removal, so
 * this catches the nodes that disappeared without an `onRemoved` call.
 */
function pruneDetachedNodes() {
  for (const node of [...controllers.keys()]) {
    if (!node || !node.graph) detach(node)
  }
}

/** Run `fn` for the controller addressed by an event's node id, if any. */
function withNode(nodeId, fn) {
  if (nodeId === undefined || nodeId === null) return
  const target = routeToController(controllers.values(), String(nodeId))
  if (target) safely(() => fn(target))
}

function forEachController(fn) {
  for (const controller of controllers.values()) safely(() => fn(controller))
}

function safely(fn) {
  try {
    fn()
  } catch (err) {
    log('preview handler failed', err)
  }
}

app.registerExtension({
  name: EXTENSION_NAME,

  async setup() {
    injectStylesheet()

    // Registering here (rather than lazily per node) is what makes ComfyUI's
    // api dispatch our custom type at all: unregistered types are dropped.
    api.addEventListener(MESSAGE_TYPE, (event) => {
      const raw = event && event.detail
      if (!raw || typeof raw !== 'object') return
      const nodeId = raw.node_id
      if (typeof nodeId !== 'string' || nodeId.length === 0) return
      const target = routeToController(controllers.values(), nodeId)
      if (!target) return
      safely(() => target.handleMessage(raw))
    })

    api.addEventListener('execution_start', () => {
      forEachController((controller) => controller.onExecutionStart())
    })

    api.addEventListener('execution_interrupted', () => {
      forEachController((controller) => controller.onInterrupted())
    })

    api.addEventListener('execution_error', (event) => {
      const detail = (event && event.detail) || {}
      withNode(detail.node_id, (controller) =>
        controller.onExecutionError(detail.exception_message || null),
      )
    })

    api.addEventListener('executed', (event) => {
      const detail = (event && event.detail) || {}
      withNode(detail.display_node || detail.node, (controller) => controller.onExecuted())
    })

    api.addEventListener('reconnecting', () => {
      forEachController((controller) => controller.setConnection('reconnecting'))
    })

    api.addEventListener('reconnected', () => {
      forEachController((controller) => controller.setConnection('online'))
    })

    // A cleared graph destroys node objects without always firing onRemoved.
    api.addEventListener('graphCleared', () => detachAll())
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!isSamplerNode(nodeData && nodeData.name, nodeData && nodeData.display_name)) {
      return
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = onNodeCreated ? onNodeCreated.apply(this, args) : undefined
      safely(() => attach(this))
      return result
    }

    const onRemoved = nodeType.prototype.onRemoved
    nodeType.prototype.onRemoved = function (...args) {
      safely(() => detach(this))
      return onRemoved ? onRemoved.apply(this, args) : undefined
    }
  },

  /** Loading another workflow replaces every node object on the canvas. */
  afterConfigureGraph() {
    safely(pruneDetachedNodes)
  },
})
