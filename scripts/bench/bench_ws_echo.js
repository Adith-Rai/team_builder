// WS echo server for measuring WS round-trip floor latency.
// Listens on port 9100; echoes any message back unchanged.
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 9100 });
wss.on('connection', ws => {
  ws.on('message', msg => ws.send(msg));
});
console.log('echo server ready on 9100');
