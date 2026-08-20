# RAVEN Streaming preview protocol (v1)

The contract between the `RAVEN Streaming Sampler` Python node and the in-node
preview widget shipped in this directory.

Everything below was checked against the pinned baseline: ComfyUI
`c67885b14556cf3e4e061862925282d403d09862` (0.33.0) and the frontend package it
pins, `comfyui-frontend-package==1.49.6` (`requirements.txt`). Line references
are to that commit and that frontend tag.

The preview is strictly an extra. **A preview failure must not affect
sampling**, and the node's real `LATENT` / `IMAGE` / `AUDIO` outputs are
produced exactly as they would be with no browser attached.

That isolation is structural, not a promise. The node runs **three lanes** off
the same per-chunk callback: a video collector and an audio collector, which are
mandatory and whose buffers *are* the `IMAGE` and `AUDIO` outputs, and this
preview lane, which sits on top of them and reads frames and PCM back out. The
preview is downstream of the outputs, so there is no path by which it can
change them. There is no whole-clip VAE decode anywhere in the node.

One consequence is visible on the wire and is documented rather than hidden:
**the audio in this stream is raw PCM, before normalisation.** The final `AUDIO`
output applies `vae_decode_audio`'s tail — `std = std(x, dim=[1,2]) * 5;
std[std < 1] = 1; x /= std` — and that divisor is a whole-clip statistic which
does not exist while the clip is still being made. A sample-for-sample
comparison of what was streamed against what was returned differs by exactly
that divisor. Video has no such split: the frames encoded here are the same
frames the `IMAGE` buffer holds.

---

## 1. What the pinned ComfyUI can actually carry

### 1.1 Binary websocket frames cannot be used

`PromptServer.send_sync(event, data, sid)` will send a binary frame when `data`
is `bytes` and `event` is an `int` (`server.py:1290-1298`, `encode_bytes` at
`server.py:1301-1308`). The event id is packed as a big-endian `uint32` header.

The frontend, however, only decodes four ids and **throws on everything else**
(`src/scripts/api.ts`, websocket `message` handler):

```js
switch (eventType) {
  case 3: /* progress_text  */ break
  case 1: /* preview image  */ break
  case 4: /* preview + meta */ break
  default:
    throw new Error(`Unknown binary websocket message of type ${eventType}`)
}
```

That `throw` is swallowed by the surrounding `try/catch`, which logs
`console.warn('Unhandled message:', ...)`. So a custom binary event is not an
error the user sees — **the payload is simply dropped**, with no way for an
extension to intercept it. The ids that *are* decoded are hardwired to image
previews and progress text (`protocol.py`), and reusing one would corrupt the
core preview UI.

There is no extension hook on the raw socket either: `ComfyApi` keeps `socket`
private to its own handlers and exposes no "raw frame" event.

**Conclusion: server → client media must travel inside JSON messages.**

### 1.2 Custom JSON message types do work, with one condition

For an unknown JSON `type`, the frontend dispatches a `CustomEvent` **only if a
listener for that exact type is already registered**:

```js
default:
  if (this._registered.has(msg.type)) {
    super.dispatchEvent(new CustomEvent(msg.type, { detail: msg.data }))
  } else if (!this.reportedUnknownMessageTypes.has(msg.type)) { ... }
```

`_registered` is filled by `api.addEventListener` / `api.addCustomEventListener`.
The extension therefore registers its listener in `setup()`, once, at startup —
never lazily when a node is created, or the first messages of a run would be
dropped.

Listener exceptions are caught by the frontend's own wrapper (`wrapListener`),
which logs at `warn` level. That is a safety net, not a design: this extension
guards its own handlers.

### 1.3 Client → server

The websocket is one-way for extensions (the client only ever sends its feature
flags, `api.ts` `open` handler). Anything the client wants to say goes over
HTTP through `api.fetchApi`, which targets `/api/...` routes registered by the
node pack. Hence the resume request in §4 is an HTTP POST, and it is optional.

---

## 2. Transport and its cost

| | |
|---|---|
| Channel | websocket JSON, `PromptServer.send_sync("raven.preview", body, sid)` |
| Message type | `raven.preview` (one type, discriminated by `event`) |
| Payload encoding | `base64` inside the JSON body |
| Framing | one JSON message per fMP4 box group (init segment, or one fragment) |

**Overhead, stated plainly.** Base64 turns `n` bytes into `ceil(n/3) * 4`
characters, i.e. **+33.3 %**, and every character is one byte in the UTF-8
websocket text frame. The base64 alphabet needs no JSON escaping, so the JSON
envelope adds only the fixed key/value text (roughly 200–300 bytes per message).
A 200 KB fragment therefore costs about 267 KB on the wire plus the envelope.
On top of that, both sides pay a base64 encode/decode pass; in the browser that
is `atob` plus a byte-copy loop, which is linear and small next to decoding the
video itself.

That cost is accepted because the alternative (§1.1) does not work at all on
the pinned frontend.

**Size guidance.** Keep a single message under ~256 KB of raw media (~342 KB
encoded). aiohttp does not limit outbound frame size, but one enormous message
blocks the socket for every other node's progress traffic while it is written.
Prefer more, smaller fragments.

**Out-of-band bodies are not part of v1.** The obvious way to avoid base64 is
to have the backend register an HTTP route and send a URL instead of bytes. It
is left out on purpose: it needs a backend route this pack does not have, and
it turns a strictly ordered stream into N concurrent fetches whose completion
order has to be re-imposed anyway. The client rejects an `init` or `segment`
without `data`. If the base64 cost ever shows up in a measurement, that is the
change to make, as protocol v2.

---

## 3. Message format

Every message body (the `data` of the websocket JSON message) is an object with
this envelope:

| field | type | required | meaning |
|---|---|---|---|
| `v` | int | yes | protocol version, currently `1`. A different value is dropped by the client. |
| `event` | string | yes | `open` \| `init` \| `segment` \| `status` \| `end` |
| `session_id` | string | yes | new for every sampler execution; see §5 |
| `node_id` | string | yes | the node's hidden `unique_id`, as a string |
| `prompt_id` | string | no | the executing prompt id |
| `seq` | int | yes | 0-based, +1 for **every** message of the session, control messages included |
| `t` | float | no | server timestamp, seconds |

`seq` is what makes ordering, de-duplication and gap detection uniform: the
client does not have to reason about which event kinds are ordered relative to
which others.

### 3.1 `open` — start of a stream

Must be the first message of a session (`seq` = the session's baseline, and the
client resets its sequencer to that value).

| field | type | required | meaning |
|---|---|---|---|
| `mime` | string | yes | full MSE type, e.g. `video/mp4; codecs="avc1.640028,mp4a.40.2"` |
| `width` / `height` | int | no | pixel size; drives the widget's aspect ratio |
| `fps` | float | no | video frame rate |
| `audio` | object \| null | no | `{ "sample_rate": 48000, "channels": 2 }`, or `null` for video-only |
| `duration_hint` | float | no | expected total seconds |
| `resync` | bool | no | `true` when the backend is restarting a stream the client already saw |
| `label` | string | no | short free text shown under the player |

`mime` must be a `video/mp4` type **with** a `codecs` parameter; the client
rejects anything else, because `MediaSource.isTypeSupported` is unreliable
without codecs and a wrong guess fails at append time instead of at setup.

### 3.2 `init` — the initialisation segment

`ftyp` + `moov`, exactly once per `open`.

| field | type | required | meaning |
|---|---|---|---|
| `encoding` | string | yes | `base64` |
| `data` | string | yes | base64 of the init segment |
| `bytes` | int | no | decoded length; the client verifies it and drops a mismatch |

### 3.3 `segment` — one media fragment

`moof` + `mdat`, in decode order.

| field | type | required | meaning |
|---|---|---|---|
| `encoding` | string | yes | `base64` |
| `data` | string | yes | base64 of the fragment |
| `bytes` | int | no | decoded length, verified |
| `index` | int | no | fragment index inside the session (diagnostics only) |
| `keyframe` | bool | no | `true` when the fragment starts at an IDR and is independently decodable |
| `start` | float | no | presentation start, seconds |
| `duration` | float | no | fragment duration, seconds |

Fragments must be appendable in the order sent — that is exactly what
`raven_streaming/media/mp4_writer.py` already produces. The preview lane muxes
with **`frag_every_frame+empty_moov+default_base_moof`** and
`min_frag_duration=1` (`PREVIEW_MOVFLAGS` / `PREVIEW_CONTAINER_OPTIONS`), so
there is one fragment per frame and the first picture reaches the browser one
*frame* after it is muxed rather than one segment. `mp4_writer.py` also defines
a `frag_keyframe` mode (`SEGMENT_MOVFLAGS`), which is **not** what the preview
uses: measured at 1376x768, that mode holds fragment *N* until the first frame
of segment *N+1*, which is exactly the delay this protocol exists to avoid
(`../COMPATIBILITY.md`, "PyAV fragment cadence"). Init is taken with
`take_init_segment`, fragments with `take_fragments`. One forced IDR per
17-frame chunk keeps a late joiner resynchronisable.

Observed shape, from the 1376x768 integration runs: 251 fragments /
6 531 496 B at 192 frames and 473 / 11 960 218 B at 362, largest fragment
165 858 B, **0 oversize**, 0 send failures, fragment indices contiguous from 0.

### 3.4 `status` — sampler phase

| field | type | required | meaning |
|---|---|---|---|
| `phase` | string | yes | `waiting` \| `model_loading` \| `sampling` \| `finalizing` |
| `message` | string | no | one short line, shown verbatim |
| `progress` | object | no | `{ "value": 12, "max": 40 }` |

`buffering`, `live` and `reconnecting` are **client-side** states, derived from
the media pipeline and the socket. The backend never sends them.

### 3.5 `end` — terminal message

| field | type | required | meaning |
|---|---|---|---|
| `reason` | string | yes | `complete` \| `cancelled` \| `error` |
| `message` | string | no | one short line |
| `segments` | int | no | how many `segment` messages the backend sent; the client warns if it appended fewer |

Client behaviour per reason:

- `complete` — drain the append queue, then `MediaSource.endOfStream()`. The
  clip stays scrubbable.
- `cancelled` — stop immediately: drop the queue, abort the source buffer,
  detach and **revoke the object URL**, remove listeners.
- `error` — keep what already decoded, close the stream, show the message. The
  sampler's own error is reported by ComfyUI separately; this line is about the
  preview only.

---

## 4. Loss, duplicates and resume

The client keeps one sequencer per session:

- `seq` below the next expected one → duplicate or replay, dropped and counted.
- `seq` above it → held (up to 64 messages) and the session is "in a gap".
- Held backlog full → overflow; buffering alone cannot recover.

After **4 s** in a gap, or on overflow, or on the first message of a session the
client never saw the `open` for (a tab that attached mid-run), the client sends:

```
POST /api/raven_streaming/preview/resume
{ "session_id": "...", "node_id": "7", "last_seq": 41,
  "client_id": "...", "reason": "segment 42 did not arrive" }
```

`last_seq` is the last **contiguously delivered** seq: everything up to and
including it is known-good.

The route is **optional**. A `404`, `405` or `501` answer is read as "this
backend has no resume support"; the client stops asking and displays that it is
waiting for the backend to resend. If the route exists, the backend may either:

1. resend messages from `last_seq + 1` with their original `seq` values, or
2. start a fresh stream: a new `open` with `resync: true` (new `session_id`, new
   `init`, `seq` restarting), which the client treats as a clean restart.

Both are supported. Option 2 is the honest choice when the backend no longer
holds the older fragments.

A dropped websocket is handled by the frontend itself (it reconnects and fires
`reconnecting` / `reconnected`). On `reconnected` the client issues one resume
request with its `last_seq`; without a resume route it waits for the backend.

---

## 5. Session and node isolation

- `session_id` is generated per sampler execution. A message whose
  `session_id` differs from the node's current session is dropped, so a stale
  stream from a cancelled run cannot append into a new one.
- `node_id` addresses one node. The client matches `String(node.id)` exactly
  first; if nothing matches, it accepts a colon-path suffix (`"12:7"` matches
  node `7`), which is how ComfyUI addresses nodes inside subgraphs. The suffix
  rule is a client-side fallback, not an upstream guarantee.
- A node whose id is still `-1` (litegraph assigns real ids only in
  `configure`) never matches.
- A new `open` for a node that already has a session tears the old one down
  first: buffer released, object URL revoked, listeners removed.

---

## 6. What the backend must provide

1. Register the web directory, so `/extensions` serves this folder. Either
   `WEB_DIRECTORY = "web"` in the pack's `__init__.py`, or `web = "web"` under
   `[tool.comfy]` in `pyproject.toml` (`nodes.py:2270-2289`). Without one of
   them the extension is never loaded. **This pack sets `WEB_DIRECTORY = "./web"`
   in the repository root `__init__.py`**, alongside `NODE_CLASS_MAPPINGS` and
   `NODE_DISPLAY_NAME_MAPPINGS`; `[tool.comfy] web` is deliberately left unset,
   because upstream honours both and would then register this folder twice,
   under two different keys, loading every file here twice.
2. Register the sampler node under a name the client recognises:
   `RAVENStreamingSampler`, `RavenStreamingSampler` or `RAVEN Streaming
   Sampler`. Any other name containing "raven", "streaming" and "sampler"
   (case-insensitive, in the mapping key or the display name) also matches.
3. Take `unique_id` as a hidden input and use it as `node_id`.
4. Send messages with `PromptServer.instance.send_sync("raven.preview", body,
   sid)`. Pass the executing client's `sid` when it is known so other tabs are
   not fed the stream.
5. Never let a send failure interrupt sampling: the send path belongs in a
   `try/except` that logs and continues.

---

## 7. Example session

```jsonc
{"v":1,"event":"open","session_id":"c8f1","node_id":"7","seq":0,
 "mime":"video/mp4; codecs=\"avc1.640028,mp4a.40.2\"",
 "width":848,"height":480,"fps":24,"audio":{"sample_rate":48000,"channels":2}}

{"v":1,"event":"status","session_id":"c8f1","node_id":"7","seq":1,
 "phase":"model_loading"}

{"v":1,"event":"init","session_id":"c8f1","node_id":"7","seq":2,
 "encoding":"base64","bytes":1180,"data":"AAAAHGZ0eXBpc281..."}

{"v":1,"event":"status","session_id":"c8f1","node_id":"7","seq":3,
 "phase":"sampling","progress":{"value":1,"max":40}}

{"v":1,"event":"segment","session_id":"c8f1","node_id":"7","seq":4,
 "encoding":"base64","bytes":98304,"index":0,"keyframe":true,
 "start":0.0,"duration":0.875,"data":"AAAAaG1vb2Y..."}

{"v":1,"event":"status","session_id":"c8f1","node_id":"7","seq":97,
 "phase":"finalizing"}

{"v":1,"event":"end","session_id":"c8f1","node_id":"7","seq":98,
 "reason":"complete","segments":46}
```
