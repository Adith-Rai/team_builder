/**
 * battle_server.js - Minimal Pokemon Showdown websocket server for battles.
 *
 * No auth, no chat, no sqlite, no user management.
 * Uses BattleStream from pokemon-showdown and exposes a websocket interface
 * compatible with poke-env.
 *
 * Usage: node battle_server.js --port 9000
 */

'use strict';

const { WebSocketServer } = require('ws');
const http = require('http');
const { BattleStream, getPlayerStreams } = require('pokemon-showdown/dist/sim');

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------
const args = process.argv.slice(2);
let port = 9000;
for (let i = 0; i < args.length; i++) {
    if (args[i] === '--port' && args[i + 1]) port = parseInt(args[i + 1], 10);
}

function log(...a) { process.stderr.write(`[battle_server] ${a.join(' ')}\n`); }

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
/** @type {Map<string, {ws: WebSocket, team: string|null, displayName: string}>} toId(name) -> connection */
const users = new Map();

/** @type {Map<string, {target: string, format: string}>} challenger name -> pending */
const pendingChallenges = new Map();

let battleCounter = 0;

/** @type {Map<string, {stream: BattleStream, players: any, p1: string, p2: string, ended: boolean}>} */
const battles = new Map();

/** @type {Map<WebSocket, string>} ws -> username */
const wsBySocket = new Map();

/** @type {Map<string, Set<string>>} username -> set of battle tags */
const userBattles = new Map();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sendTo(ws, msg) {
    if (ws.readyState === 1) ws.send(msg);
}

function sendToUser(name, msg) {
    const u = users.get(name);
    if (u) sendTo(u.ws, msg);
}

function toId(name) {
    return ('' + name).toLowerCase().replace(/[^a-z0-9]/g, '');
}

// ---------------------------------------------------------------------------
// Battle management
// ---------------------------------------------------------------------------

function startBattle(p1Name, p2Name, format, p1Team, p2Team) {
    battleCounter++;
    const tag = `battle-${format}-${battleCounter}`;

    // Use display names for BattleStream so poke-env can match usernames (case-sensitive)
    const p1Display = users.get(p1Name) ? users.get(p1Name).displayName : p1Name;
    const p2Display = users.get(p2Name) ? users.get(p2Name).displayName : p2Name;

    const stream = new BattleStream({ keepAlive: false });
    const players = getPlayerStreams(stream);
    const entry = { stream, players, p1: p1Name, p2: p2Name, p1Display, p2Display, ended: false };
    battles.set(tag, entry);

    // Track which users are in which battles
    if (!userBattles.has(p1Name)) userBattles.set(p1Name, new Set());
    if (!userBattles.has(p2Name)) userBattles.set(p2Name, new Set());
    userBattles.get(p1Name).add(tag);
    userBattles.get(p2Name).add(tag);

    // Pump player streams -> websocket
    pumpPlayer(tag, 'p1', p1Name, players.p1, entry);
    pumpPlayer(tag, 'p2', p2Name, players.p2, entry);

    // Start the battle — use display names so Showdown emits them in |player| messages
    stream.write(`>start ${JSON.stringify({ formatid: format })}`);
    stream.write(`>player p1 ${JSON.stringify({ name: p1Display, team: p1Team || '' })}`);
    stream.write(`>player p2 ${JSON.stringify({ name: p2Display, team: p2Team || '' })}`);

    log(`Battle started: ${tag} (${p1Display} vs ${p2Display})`);
    return tag;
}

async function pumpPlayer(tag, slot, userName, playerStream, entry) {
    try {
        let first = true;
        let chunk;
        while ((chunk = await playerStream.read()) !== null) {
            if (entry.ended) break;

            // Drain all immediately-available chunks and combine them.
            // BattleStream writes multiple chunks synchronously (gametype, player,
            // request, start+switches), and poke-env expects them in one message
            // so that |request| (which sets player_role) arrives before |switch|.
            let combined = chunk;
            while (playerStream.buf.length > 0) {
                const extra = await playerStream.read();
                if (extra === null) break;
                combined += '\n' + extra;
            }

            if (first) {
                // Prepend |init|battle and |title| so poke-env creates the battle object
                combined = `|init|battle\n|title|${entry.p1Display} vs. ${entry.p2Display}\n${combined}`;
                first = false;
            }
            // Send with battle room prefix, exactly as Showdown does
            sendToUser(userName, `>${tag}\n${combined}`);

            // Detect battle end from |win| or |tie| in the pumped chunks
            // and schedule cleanup after a short delay to let final messages flow.
            if (!entry.ended) {
                const lines = combined.split('\n');
                for (const line of lines) {
                    if (line.startsWith('|win|') || line === '|tie' || line.startsWith('|tie|')) {
                        if (!entry._cleanupScheduled) {
                            entry._cleanupScheduled = true;
                            log(`Battle end detected in ${tag}, scheduling cleanup`);
                            setTimeout(() => cleanupBattle(tag), 5000);
                        }
                        break;
                    }
                }
            }
        }
    } catch (err) {
        if (!entry.ended) log(`${slot} stream error in ${tag}: ${err.message}`);
    }
}

function cleanupBattle(tag) {
    const entry = battles.get(tag);
    if (!entry) return;
    entry.ended = true;
    try { entry.stream.destroy(); } catch (_) {}
    battles.delete(tag);
    // Remove from user tracking
    for (const name of [entry.p1, entry.p2]) {
        const s = userBattles.get(name);
        if (s) { s.delete(tag); if (s.size === 0) userBattles.delete(name); }
    }
    log(`Cleaned up ${tag} (${battles.size} active)`);
}

// ---------------------------------------------------------------------------
// Message handling
// ---------------------------------------------------------------------------

function handleMessage(ws, raw) {
    const name = wsBySocket.get(ws);
    const msg = raw.toString().trim();
    if (!msg) return;

    // Battle choice: "battle-xxx|/choose move 1" or "battle-xxx|/leave"
    const pipeIdx = msg.indexOf('|');
    if (pipeIdx > 0 && msg.startsWith('battle-')) {
        const tag = msg.slice(0, pipeIdx);
        const cmd = msg.slice(pipeIdx + 1).trim();
        const entry = battles.get(tag);
        if (!entry) return;

        if (cmd.startsWith('/leave')) {
            // no-op, battle cleans up on |win|
            return;
        }
        if (cmd.startsWith('/choose ')) {
            const choice = cmd.slice(8);
            const slot = (name === entry.p1) ? 'p1' : 'p2';
            entry.players[slot].write(choice);
            return;
        }
        if (cmd.startsWith('/team ')) {
            // Team preview lead order: /team 123456
            const teamOrder = cmd.slice(1); // keep "team 123456"
            const slot = (name === entry.p1) ? 'p1' : 'p2';
            entry.players[slot].write(teamOrder);
            return;
        }
        if (cmd === '/timer on' || cmd === '/timer off') return;
        return;
    }

    // Global commands (prefixed with |)
    const cmdBody = msg.startsWith('|') ? msg.slice(1) : msg;

    if (cmdBody.startsWith('/trn ')) {
        // Login: /trn USERNAME,0,ASSERTION
        const parts = cmdBody.slice(5).split(',');
        const loginName = parts[0].trim();
        const id = toId(loginName);

        // Remove old mapping if re-logging under a different name
        if (name && name !== id) {
            users.delete(name);
        }
        // If another connection already has this username, kick it
        const existing = users.get(id);
        if (existing && existing.ws !== ws && existing.ws.readyState === 1) {
            log(`Kicking existing connection for ${id} (replaced by new login)`);
            try { existing.ws.close(); } catch (_) {}
        }
        wsBySocket.set(ws, id);
        users.set(id, { ws, team: (existing || {}).team || null, displayName: loginName });

        // Send updateuser + challstr (poke-env needs challstr first to trigger login,
        // but since we already got /trn, just confirm with updateuser)
        sendTo(ws, `|updateuser| ${loginName}|1|`);
        log(`Login: ${id}`);
        return;
    }

    if (cmdBody.startsWith('/autojoin')) {
        // no-op
        return;
    }

    if (cmdBody.startsWith('/utm ')) {
        const teamData = cmdBody.slice(5).trim();
        if (name && users.has(name)) {
            users.get(name).team = (teamData === 'null') ? null : teamData;
        }
        return;
    }

    if (cmdBody.startsWith('/challenge ')) {
        // /challenge OPPONENT, FORMAT
        const rest = cmdBody.slice(11);
        const commaIdx = rest.indexOf(',');
        if (commaIdx < 0) return;
        const target = toId(rest.slice(0, commaIdx).trim());
        const format = toId(rest.slice(commaIdx + 1).trim());

        pendingChallenges.set(name, { target, format });

        // Notify the target about the challenge via updatechallenges
        sendToUser(target, `|updatechallenges|{"challengesFrom":{"${name}":"${format}"},"challengeTo":null}`);
        // Also send pm-based challenge notification (poke-env checks both paths)
        sendToUser(target, `|pm| ${name}| ${target}|/challenge ${format}`);
        log(`Challenge: ${name} -> ${target} (${format})`);
        return;
    }

    if (cmdBody.startsWith('/accept ')) {
        const challenger = toId(cmdBody.slice(8).trim());
        const challenge = pendingChallenges.get(challenger);
        if (!challenge || challenge.target !== name) {
            log(`No pending challenge from ${challenger} for ${name}`);
            return;
        }
        pendingChallenges.delete(challenger);

        const p1Team = users.get(challenger) ? users.get(challenger).team : null;
        const p2Team = users.get(name) ? users.get(name).team : null;

        startBattle(challenger, name, challenge.format, p1Team, p2Team);
        return;
    }

    if (cmdBody.startsWith('/search ') || cmdBody === '/timer on' || cmdBody === '/timer off') {
        return; // ignore
    }

    if (cmdBody.startsWith('/leave ')) {
        return; // ignore
    }

    log(`Unhandled message from ${name}: ${msg}`);
}

// ---------------------------------------------------------------------------
// Server setup
// ---------------------------------------------------------------------------

const server = http.createServer((req, res) => {
    // action.php endpoint for poke-env login (no-op, return guest assertion)
    if (req.url && req.url.startsWith('/action.php')) {
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end(']{"actionsuccess":true,"assertion":""}');
        return;
    }
    res.writeHead(200);
    res.end('battle_server ok');
});

const wss = new WebSocketServer({ server, path: '/showdown/websocket' });

wss.on('connection', (ws) => {
    const guestId = `guest${Math.floor(Math.random() * 1e9)}`;
    wsBySocket.set(ws, guestId);
    log(`Connection opened (${guestId})`);

    // Send challstr so poke-env triggers its login flow
    sendTo(ws, `|challstr|0|${guestId}`);

    ws.on('message', (data) => {
        try {
            handleMessage(ws, data);
        } catch (err) {
            log(`Error handling message: ${err.stack}`);
        }
    });

    ws.on('close', () => {
        const name = wsBySocket.get(ws);
        log(`Connection closed (${name})`);
        wsBySocket.delete(ws);
        if (name) {
            users.delete(name);
            pendingChallenges.delete(name);
            // Clean up any active battles for this user
            const bset = userBattles.get(name);
            if (bset) {
                for (const tag of bset) {
                    const entry = battles.get(tag);
                    if (entry && !entry.ended) {
                        // Force win for the other player
                        const other = (entry.p1 === name) ? 'p2' : 'p1';
                        try { entry.stream.write(`>forcewin ${other}`); } catch (_) {}
                    }
                }
            }
        }
    });
});

server.listen(port, '0.0.0.0', () => {
    log(`Listening on port ${port} (ws://127.0.0.1:${port}/showdown/websocket)`);
});
