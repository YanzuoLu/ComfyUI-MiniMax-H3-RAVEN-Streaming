/**
 * Which node a message belongs to, and which node type gets a preview.
 *
 * Pure module - no DOM, no ComfyUI imports, no import-time side effects.
 *
 * Two separate identity problems live here:
 *
 * 1. Node *type* matching. The Python side of this pack registers the sampler
 *    under a mapping key that is not frozen yet, so the extension accepts the
 *    known spellings instead of hardcoding one guess. `PROTOCOL.md` states this
 *    as a contract the backend has to satisfy.
 *
 * 2. Node *instance* matching. The backend sends the `unique_id` it received
 *    as a hidden input. On the canvas the same node is `node.id`. Inside a
 *    subgraph, ComfyUI's execution ids are colon-joined paths ("12:7:3"), so an
 *    exact match is tried first and a path-suffix match second. The suffix rule
 *    is a client-side heuristic, not an upstream guarantee - it is only reached
 *    when the exact match fails.
 */

/** Node type names this extension attaches a preview to. */
export const SAMPLER_NODE_NAMES = Object.freeze([
  'RAVENStreamingSampler',
  'RavenStreamingSampler',
  'RAVEN Streaming Sampler',
])

/** Fallback: every token must appear in the name (case-insensitive). */
const NAME_TOKENS = Object.freeze(['raven', 'streaming', 'sampler'])

function normalise(value) {
  return String(value == null ? '' : value)
    .toLowerCase()
    .replace(/[\s_-]+/g, '')
}

/**
 * @param {string} nodeName  the NODE_CLASS_MAPPINGS key
 * @param {string} [displayName]
 */
export function isSamplerNode(nodeName, displayName) {
  const candidates = [nodeName, displayName].filter(Boolean)
  if (candidates.length === 0) return false

  const normalisedNames = SAMPLER_NODE_NAMES.map(normalise)
  for (const candidate of candidates) {
    if (normalisedNames.includes(normalise(candidate))) return true
  }
  for (const candidate of candidates) {
    const lower = String(candidate).toLowerCase()
    if (NAME_TOKENS.every((token) => lower.includes(token))) return true
  }
  return false
}

/**
 * Does an event's `node_id` address this canvas node?
 *
 * @param {string|number} canvasId  `node.id` (may be -1 before configure)
 * @param {string} eventNodeId      `node_id` from the wire
 */
export function matchesNode(canvasId, eventNodeId) {
  if (canvasId === undefined || canvasId === null) return false
  if (typeof eventNodeId !== 'string' || eventNodeId.length === 0) return false
  const id = String(canvasId)
  // A node that has not been configured yet has id -1 and cannot own a stream.
  if (id === '-1') return false
  if (id === eventNodeId) return true
  return eventNodeId.endsWith(`:${id}`)
}

/**
 * Pick one controller out of many for an incoming event.
 * Exact id matches win over subgraph-suffix matches, so a top-level node with
 * id "3" is never fed a stream belonging to "12:3".
 *
 * @param {Iterable<{nodeId: string|number}>} controllers
 * @param {string} eventNodeId
 */
export function routeToController(controllers, eventNodeId) {
  let suffixMatch = null
  for (const controller of controllers) {
    const id = String(controller.nodeId)
    if (id === '-1') continue
    if (id === eventNodeId) return controller
    if (suffixMatch === null && eventNodeId.endsWith(`:${id}`)) {
      suffixMatch = controller
    }
  }
  return suffixMatch
}
