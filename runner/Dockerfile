FROM node:24-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends git python3 python3-venv python3-pip ca-certificates \
  && rm -rf /var/lib/apt/lists/* \
  && useradd -u 10001 -m runner

WORKDIR /app
COPY package.json ./
COPY dist ./dist
COPY runner ./runner
RUN npm install --omit=dev && chown -R runner /app

USER runner
ENTRYPOINT ["node", "runner/agent.mjs"]
CMD ["run"]
