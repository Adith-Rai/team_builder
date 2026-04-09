FROM node:20-alpine
WORKDIR /showdown
RUN apk add --no-cache git curl
RUN git clone --depth 1 https://github.com/smogon/pokemon-showdown.git .
RUN npm ci --omit=dev
EXPOSE 8000
CMD ["node","pokemon-showdown","start","--no-security","--host","0.0.0.0","--port","8000"]
