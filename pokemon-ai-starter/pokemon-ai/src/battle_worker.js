/**
 * battle_worker.js - Manages concurrent Pokemon battles via BattleStream.
 *
 * Communicates with Python over JSON lines on stdin/stdout.
 * Debug logs go to stderr so they never mix with protocol data.
 *
 * Protocol (Python -> Node, stdin):
 *   {"type":"start","id":"b1","format":"gen9ou","p1_team":"PACKED","p2_team":"PACKED","seed":[1,2,3,4]}
 *   {"type":"choose","id":"b1","player":"p1","choice":"move 1"}
 *   {"type":"forfeit","id":"b1","player":"p1"}
 *
 * Protocol (Node -> Python, stdout):
 *   {"type":"update","id":"b1","messages":[...]}
 *   {"type":"sideupdate","id":"b1","player":"p1","messages":[...]}
 *   {"type":"end","id":"b1","winner":"Bot1"}
 *   {"type":"error","id":"b1","message":"..."}
 */

'use strict';

const readline = require('readline');
const { BattleStream, getPlayerStreams, Teams } = require('pokemon-showdown/dist/sim');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function debug(...args) {
    process.stderr.write('[battle_worker] ' + args.join(' ') + '\n');
}

function send(obj) {
    process.stdout.write(JSON.stringify(obj) + '\n');
}

// ---------------------------------------------------------------------------
// Battle registry
// ---------------------------------------------------------------------------

/** @type {Map<string, {stream: BattleStream, players: ReturnType<typeof getPlayerStreams>, ended: boolean}>} */
const battles = new Map();

// ---------------------------------------------------------------------------
// Start a new battle
// ---------------------------------------------------------------------------

function startBattle(msg) {
    const { id, format, p1_team, p2_team, seed, p1_name, p2_name } = msg;

    if (battles.has(id)) {
        send({ type: 'error', id, message: `Battle ${id} already exists` });
        return;
    }

    try {
        const stream = new BattleStream({ keepAlive: false });
        const players = getPlayerStreams(stream);
        const entry = { stream, players, ended: false, _activityTimer: null };
        battles.set(id, entry);

        // Initial inactivity timeout — if no stream activity within 120s, force-end
        entry._activityTimer = setTimeout(() => {
            if (!entry.ended) {
                debug(`Battle ${id} timed out (no activity for 120s)`);
                entry.ended = true;
                send({ type: 'error', id, message: 'Battle timed out (120s inactivity)' });
                send({ type: 'end', id, winner: null });
                cleanupBattle(id);
            }
        }, 120000);

        // Wire up output forwarding for each player stream (p1, p2).
        for (const pKey of ['p1', 'p2']) {
            pumpPlayerStream(id, pKey, players[pKey], entry);
        }

        // Wire up the omniscient stream to detect |win| / |tie|.
        pumpOmniscientStream(id, players.omniscient, entry);

        // Build start options
        const startOpts = { formatid: format };
        if (seed) startOpts.seed = seed;

        const name1 = p1_name || 'p1';
        const name2 = p2_name || 'p2';

        stream.write(`>start ${JSON.stringify(startOpts)}`);
        stream.write(`>player p1 ${JSON.stringify({ name: name1, team: p1_team })}`);
        stream.write(`>player p2 ${JSON.stringify({ name: name2, team: p2_team })}`);

        debug(`Started battle ${id} (${format})`);
    } catch (err) {
        send({ type: 'error', id, message: err.message });
        battles.delete(id);
    }
}

// ---------------------------------------------------------------------------
// Async stream pumps
// ---------------------------------------------------------------------------

async function pumpPlayerStream(id, player, playerStream, entry) {
    try {
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

            // Reset inactivity timeout on each chunk
            if (entry._activityTimer) clearTimeout(entry._activityTimer);
            entry._activityTimer = setTimeout(() => {
                if (!entry.ended) {
                    debug(`Battle ${id} timed out (no activity for 120s)`);
                    entry.ended = true;
                    send({ type: 'error', id, message: 'Battle timed out (120s inactivity)' });
                    send({ type: 'end', id, winner: null });
                    cleanupBattle(id);
                }
            }, 120000);

            // Send the raw chunk exactly as poke-env would receive it from
            // a websocket. Prepend the battle tag so the Python side can
            // process it with the same _handle_battle_message logic.
            const rawMessage = `>${id}\n${combined}`;
            send({ type: 'sideupdate', id, player, raw: rawMessage });
        }
    } catch (err) {
        if (!entry.ended) {
            send({ type: 'error', id, message: `${player} stream error: ${err.message}` });
        }
    }
}

async function pumpOmniscientStream(id, omniStream, entry) {
    try {
        let chunk;
        while ((chunk = await omniStream.read()) !== null) {
            if (entry.ended) break;

            // Detect end-of-battle from omniscient stream
            const lines = chunk.split('\n');
            for (const line of lines) {
                if (line.startsWith('|win|')) {
                    const winner = line.slice(5);
                    entry.ended = true;
                    send({ type: 'end', id, winner });
                    cleanupBattle(id);
                    return;
                }
                if (line === '|tie' || line.startsWith('|tie|')) {
                    entry.ended = true;
                    send({ type: 'end', id, winner: null });
                    cleanupBattle(id);
                    return;
                }
            }
            // We don't send omniscient updates to Python — the per-player
            // streams already contain all the info each side needs.
        }
    } catch (err) {
        if (!entry.ended) {
            send({ type: 'error', id, message: `omniscient stream error: ${err.message}` });
        }
    }
}

// ---------------------------------------------------------------------------
// Choose / Forfeit
// ---------------------------------------------------------------------------

function handleChoose(msg) {
    const { id, player, choice } = msg;
    const entry = battles.get(id);
    if (!entry) {
        send({ type: 'error', id, message: `Battle ${id} does not exist` });
        return;
    }
    if (entry.ended) {
        send({ type: 'error', id, message: `Battle ${id} already ended` });
        return;
    }

    try {
        const pStream = entry.players[player];
        if (!pStream) {
            send({ type: 'error', id, message: `Invalid player: ${player}` });
            return;
        }
        pStream.write(choice);
    } catch (err) {
        send({ type: 'error', id, message: `Choose error: ${err.message}` });
    }
}

function handleForfeit(msg) {
    const { id, player } = msg;
    const entry = battles.get(id);
    if (!entry) {
        send({ type: 'error', id, message: `Battle ${id} does not exist` });
        return;
    }
    if (entry.ended) return;

    try {
        // Writing ">forfeit" to the main battle stream causes the other player to win.
        entry.stream.write(`>forcewin ${player === 'p1' ? 'p2' : 'p1'}`);
    } catch (err) {
        send({ type: 'error', id, message: `Forfeit error: ${err.message}` });
    }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

function cleanupBattle(id) {
    const entry = battles.get(id);
    if (!entry) return;
    entry.ended = true;
    if (entry._activityTimer) {
        clearTimeout(entry._activityTimer);
        entry._activityTimer = null;
    }
    try {
        entry.stream.destroy();
    } catch (_) { /* ignore */ }
    battles.delete(id);
    debug(`Cleaned up battle ${id} (${battles.size} active)`);
}

// ---------------------------------------------------------------------------
// Stdin message dispatch
// ---------------------------------------------------------------------------

const rl = readline.createInterface({
    input: process.stdin,
    terminal: false,
});

rl.on('line', (line) => {
    line = line.trim();
    if (!line) return;

    let msg;
    try {
        msg = JSON.parse(line);
    } catch (err) {
        debug(`Invalid JSON: ${line}`);
        send({ type: 'error', id: null, message: `Invalid JSON: ${err.message}` });
        return;
    }

    try {
        switch (msg.type) {
            case 'start':
                startBattle(msg);
                break;
            case 'choose':
                handleChoose(msg);
                break;
            case 'forfeit':
                handleForfeit(msg);
                break;
            default:
                send({ type: 'error', id: msg.id || null, message: `Unknown message type: ${msg.type}` });
        }
    } catch (err) {
        debug(`Unhandled error processing message: ${err.stack}`);
        send({ type: 'error', id: msg.id || null, message: `Internal error: ${err.message}` });
    }
});

rl.on('close', () => {
    debug('stdin closed, shutting down');
    // Clean up all battles
    for (const id of battles.keys()) {
        cleanupBattle(id);
    }
    process.exit(0);
});

// Keep process alive
process.stdin.resume();

debug(`Ready (pid=${process.pid})`);
send({ type: 'ready' });
