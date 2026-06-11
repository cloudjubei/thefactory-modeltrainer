# thefactory compute runner agent

Runs training jobs dispatched by the Overseer on this machine. Pairing is a one-time PIN
exchange; afterwards the agent long-polls the backend with its own token (stored hashed
server-side), executes jobs, and streams logs + results back. Declared data files are
fetched once into a content-addressed cache (`~/.thefactory-runner/cache`) — re-runs that
need the same files download nothing.

## Setup (bare, same or another machine)

```bash
# from a checkout of thefactory-modeltrainer (npm install once for thefactory-tools)
node runner/agent.mjs pair --backend http://<overseer-host>:7001 --name gpu-box
# enter the PIN from Overseer → Settings → Compute Runners → "Pair a new runner"
node runner/agent.mjs run
```

Target it from the Model Trainer app: set "Run on" in the Launch tab to the runner's id
(shown in Settings → Compute Runners).

## Docker

```bash
docker build -t thefactory/runner -f runner/Dockerfile .
docker run -it -v thefactory-runner:/home/runner/.thefactory-runner thefactory/runner \
  pair --backend http://host.docker.internal:7001 --name docker-box
docker run -d -v thefactory-runner:/home/runner/.thefactory-runner \
  -v /path/to/projects:/path/to/projects thefactory/runner run
```

The volume keeps the pairing + data cache across container restarts.

## Current limits (v1, by design)

- Jobs reference projects by **local path** — the path must exist on the runner's machine
  (same machine, a shared mount, or bind-mounted into the container at the same path).
  `git` repo refs clone into `~/.thefactory-runner/workspaces` but the cloned project must
  be self-bootstrapping (the trainer venv convention assumes a prepared checkout).
- One job at a time; abort is honoured between polls and kills the running child.
