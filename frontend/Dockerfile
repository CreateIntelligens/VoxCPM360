FROM node:22-alpine AS builder

WORKDIR /frontend
RUN corepack enable && corepack install --global pnpm@11.18.0
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

FROM nginx:alpine AS runner
COPY nginx.template /etc/nginx/templates/default.conf.template
COPY --from=builder /frontend/dist /usr/share/nginx/html
