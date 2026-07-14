// App↔Overseer bridge client. When this app runs embedded in Overseer's App
// view, `window.OverseerBridge` lets it read/write its own DataStorage records
// through the host — the host holds the write credential, this app never does.
// Standalone (opened directly, not in an iframe) `embedded` is false and the
// app falls back to localStorage.
;(function () {
  const PREFIX = 'overseer:'
  const pending = new Map()
  let seq = 0

  window.addEventListener('message', (event) => {
    const data = event.data
    if (!data || data.overseerBridgeResponse !== true) return
    const entry = pending.get(data.id)
    if (!entry) return
    pending.delete(data.id)
    clearTimeout(entry.timer)
    if (data.ok) entry.resolve(data.result)
    else entry.reject(new Error(data.error || 'Overseer bridge error'))
  })

  function call(type, payload, timeoutMs) {
    return new Promise((resolve, reject) => {
      const id = 'req-' + ++seq
      const timer = setTimeout(() => {
        if (pending.delete(id)) reject(new Error('Overseer bridge timeout: ' + type))
      }, timeoutMs || 8000)
      pending.set(id, { resolve, reject, timer })
      window.parent.postMessage({ type: PREFIX + type, id: id, payload: payload }, '*')
    })
  }

  window.OverseerBridge = {
    embedded: window.parent && window.parent !== window,
    // `timeoutMs` overrides the default 8s guard — bulk history scans page a large project under DB write
    // contention, where a single page can legitimately take longer than 8s (a too-tight timeout rejects,
    // which the accumulator would otherwise mistake for end-of-data and truncate the scan).
    queryData: (payload, timeoutMs) => call('data.query', payload, timeoutMs),
    // Total records matching a filter (type/where), ignoring limit/offset — pairs with queryData's
    // limit/offset for server-side pagination. Returns { count }.
    countData: (payload) => call('data.count', payload),
    putData: (payload) => call('data.put', payload),
    deleteData: (payload) => call('data.delete', payload),
    // The records of every live-data source this project is subscribed to.
    readLiveData: () => call('live-data.read', undefined),
    // Layered settings: a user-global default (shared across every app) plus an
    // optional per-app override. `getSettings` returns both raw layers
    // ({ global, app }); resolve them as `app ?? global ?? inferred`.
    getSettings: (key) => call('settings.get', { key: key }),
    putSettings: (key, level, value) =>
      call('settings.put', { key: key, level: level, value: value }),
    // Clear this app's override so the user-global default applies again.
    clearAppSetting: (key) => call('settings.delete', { key: key }),
    // Run a named analysis job (web search + LLM); it writes records this app
    // re-reads via data.query. Slow (often >90s), so allow 3 minutes. Even if
    // this call times out, the backend still writes the record — callers re-read.
    runJob: (jobName, params) => call('analysis.run', { jobName: jobName, params: params }, 180000),
    // Start a DETACHED background activity. Returns { activityId } immediately;
    // the activity keeps running server-side even if this app is closed or the
    // user navigates away. Observe progress + results by re-reading records via
    // data.query, and watch the nav-panel activity spinner.
    startActivity: (activityType, params) =>
      call('activities.start', { activityType: activityType, params: params }, 30000),
    listActivities: () => call('activities.list', undefined),
    abortActivity: (activityId) => call('activities.abort', { activityId: activityId }),
    // Resume a run that's no longer live (e.g. orphaned by a server restart).
    // No-op if it's still genuinely running. Returns { activityId }.
    resumeActivity: (activityId) => call('activities.resume', { activityId: activityId }, 30000),
    // Compute runners (remote machines that run training jobs). Pairing returns
    // a short-lived { pin, pairingId, expiresAt } to enter on the runner; the
    // runner's token is minted + stored hashed server-side, never here.
    createRunnerPairing: () => call('runners.create-pairing', undefined),
    listRunners: () => call('runners.list', undefined),
    removeRunner: (runnerId) => call('runners.remove', { runnerId: runnerId }),
    // Open the host's docked chat sidebar (no seed). The host decides which
    // project topic to attach.
    requestChatSidebar: () => call('chat.requestSidebar', undefined, 30000),
    // Open the docked chat sidebar seeded with a topic + first message, and
    // auto-send the seed. Used by "Ask AI for help" on a failed run.
    discussTopic: (payload) => call('chat.discuss', payload, 30000),
    // Create a story in the ACTIVE project ({ title, description } → { storyId }) — the "work on this"
    // seam (e.g. a paper coverage-gap becomes a story agents can pick up from the Stories screen).
    createStory: (payload) => call('story.create', payload, 30000),
    // Create a FEATURE inside a find-or-create story (matched by title):
    // { storyTitle, storyDescription, feature: { title, description } } → { storyId, featureId }.
    // For recurring buckets (e.g. "implement missing model components") that collect features over time.
    createStoryFeature: (payload) => call('story.feature.create', payload, 30000),
    // Tell the host which model kinds this app's background activities can run on, so the activity
    // model chip can disable unsupported options (e.g. CLI for API-only judge/propose/analyze).
    reportCapabilities: (payload) => call('app.capabilities', payload),
  }

  // The trainer's LLM activities (judge/propose/analyze-paper/consolidate/xai-narrate/…) run API-only
  // (`requiresApi`) — EXCEPT `research-training-papers` (Papers → Research papers), whose deep-research
  // discovery + verify + synthesis can run on the more capable CLI agents. Declare both once on load so
  // the host offers CLI in the activity chip and sends the CLI model for JUST that launch, while every
  // other trainer activity still runs API-only. Fire-and-forget — a failure just leaves CLI enabled.
  if (window.OverseerBridge.embedded) {
    window.OverseerBridge.reportCapabilities({
      activitiesApiOnly: true,
      cliActivities: ['research-training-papers'],
    }).catch(function () {})
  }
})()
