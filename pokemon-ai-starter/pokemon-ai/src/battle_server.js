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

function log(...a) {
    const t = new Date().toISOString().slice(11, 23); // HH:MM:SS.mmm
    process.stderr.write(`[battle_server ${t}] ${a.join(' ')}\n`);
}

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
    if (u) {
        sendTo(u.ws, msg);
        if (process.env.BS_TRACE_USER && process.env.BS_TRACE_USER === name) {
            process.stderr.write(`[BS-TX:${name}] ${msg.slice(0, 200)}\n`);
        }
    } else if (process.env.BS_TRACE_MISS) {
        process.stderr.write(`[BS-MISS] sendToUser to absent name=${name}\n`);
    }
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

// Showdown-faithful clients: usernames that need real-Showdown's per-event
// frame layout instead of poke-env 0.10's "everything bundled" layout. Their
// protocol parsers (Foul Play's fp/run_battle.py, Metamon's poke-env fork)
// hang or KeyError when given the bundled form. We match against the *display*
// name (with dashes intact), since the toId form strips the `MM-` dash. The
// `mm` ID-form prefix is also accepted as a fallback for resilience.
function isShowdownFaithful(displayOrId) {
    if (!displayOrId) return false;
    const lc = displayOrId.toLowerCase();
    if (lc.startsWith('foulplay')) return true;
    if (lc.startsWith('metamon')) return true;
    if (lc.startsWith('mm-')) return true;       // display: MM-Minikazam, MM-SmallRL, etc.
    if (/^mm[a-z]/.test(lc)) return true;        // toId: mmminikazam, mmsmallrl
    return false;
}

// Inject a monotonic rqid into the JSON of a |request| line. BattleStream's
// |request| omits rqid; Foul Play's user_json[constants.RQID] crashes without
// it. poke-env 0.10 doesn't use rqid for anything stateful.
function injectRqid(line, entry) {
    const prefix = '|request|';
    if (!line.startsWith(prefix)) return line;
    const json = line.slice(prefix.length);
    if (!json || /"rqid":/.test(json)) return line;
    try {
        const parsed = JSON.parse(json);
        entry._rqidCounter = (entry._rqidCounter || 0) + 1;
        parsed.rqid = entry._rqidCounter;
        return prefix + JSON.stringify(parsed);
    } catch (_) {
        return line;
    }
}

async function pumpPlayer(tag, slot, userName, playerStream, entry) {
    try {
        let first = true;
        let chunk;
        const faithful = isShowdownFaithful(userName);
        while ((chunk = await playerStream.read()) !== null) {
            if (entry.ended) break;

            // Drain all immediately-available BattleStream chunks into one
            // logical batch. BattleStream emits gametype/player/request/etc
            // synchronously, so a single drain captures the full event burst.
            let combined = chunk;
            while (playerStream.buf.length > 0) {
                const extra = await playerStream.read();
                if (extra === null) break;
                combined += '\n' + extra;
            }
            if (first) {
                combined = `|init|battle\n|title|${entry.p1Display} vs. ${entry.p2Display}\n${combined}`;
                first = false;
            }

            if (!faithful) {
                // poke-env 0.10 wants the whole batch as one ws frame.
                sendToUser(userName, `>${tag}\n${combined}`);
            } else {
                // Showdown-faithful layout. Required by Foul Play's parser
                // and Metamon's fork. Frame breakdown:
                //   1. |init|battle + |title|<title>   (paired, first only)
                //   2. |player|p1|<name>|              (its own frame)
                //   3. |player|p2|<name>|              (its own frame)
                //   4. main bundle: everything else (|t:|, |gametype|,
                //      |teamsize|, |gen|, |tier|, |rule|, |clearpoke|,
                //      |poke|, |teampreview|) minus |request|
                //   5. |request|<json+rqid>            (separate, last)
                //
                // Why each frame must look this way:
                //  - frame 1: FP's get_battle_tag_and_opponent reads
                //    split_msg[4] expecting the title.
                //  - frames 2/3: FP's start_battle_common loop matches on
                //    `|player|` substring AND opponent_name, then reads
                //    msg.split("|")[2] as the slot (p1/p2). If |player|
                //    is bundled with |t:|<timestamp> or |gametype|, split[2]
                //    is the timestamp/value of THAT prior field instead of
                //    the slot, and ID_LOOKUP raises KeyError.
                //  - frame 4: FP's clearpoke loop reads until the msg
                //    contains "clearpoke", then parses |poke| from
                //    msg.split("clearpoke")[-1]. Bundling clearpoke + pokes
                //    + teampreview keeps the parse working.
                //  - frame 5: FP's get_first_request_json wants
                //    split_msg[1] == "request" as a standalone msg.
                const allLines = combined.split('\n');
                const used = new Set();
                const findIdx = (pred) => allLines.findIndex(pred);

                // Frame 1: |init|battle + |title|
                const initIdx = findIdx(l => l.startsWith('|init|battle'));
                const titleIdx = findIdx(l => l.startsWith('|title|'));
                const initFrame = [];
                if (initIdx >= 0) { initFrame.push(allLines[initIdx]); used.add(initIdx); }
                if (titleIdx >= 0) { initFrame.push(allLines[titleIdx]); used.add(titleIdx); }
                if (initFrame.length > 0) {
                    sendToUser(userName, `>${tag}\n${initFrame.join('\n')}`);
                }

                // Frames 2 & 3: |player|p1|... and |player|p2|... each as own frame
                for (let i = 0; i < allLines.length; i++) {
                    if (used.has(i)) continue;
                    if (allLines[i].startsWith('|player|')) {
                        sendToUser(userName, `>${tag}\n${allLines[i]}`);
                        used.add(i);
                    }
                }

                // Frame 4: rest of the bundle (excluding |request|)
                const restLines = [];
                const requestLines = [];
                for (let i = 0; i < allLines.length; i++) {
                    if (used.has(i)) continue;
                    const ln = allLines[i];
                    if (!ln) continue;
                    if (ln.startsWith('|request|')) requestLines.push(ln);
                    else restLines.push(ln);
                }
                if (restLines.length > 0) {
                    sendToUser(userName, `>${tag}\n${restLines.join('\n')}`);
                }

                // Frame 5: |request|<json+rqid> separately
                for (const reqLine of requestLines) {
                    sendToUser(userName, `>${tag}\n${injectRqid(reqLine, entry)}`);
                }
            }

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
    const idleUsers = [];
    for (const name of [entry.p1, entry.p2]) {
        const s = userBattles.get(name);
        if (s) { s.delete(tag); if (s.size === 0) { userBattles.delete(name); idleUsers.push(name); } }
        else { idleUsers.push(name); }
    }
    log(`Cleaned up ${tag} (${battles.size} active)`);

    // Multi-battle subprocess flow: if a /challenge was issued while the
    // target was still in this battle, the |pm| /challenge was consumed by
    // the target's `pokemon_battle` loop (which silently swallows /pm
    // messages — there's no handler for them in mid-battle parsing). When
    // the target loops back to `accept_challenge` and waits for /pm, the
    // original is already gone and the loop hangs forever. Resend the |pm|
    // for any pending challenges targeted at users who just became idle so
    // their fresh accept loop picks them up.
    for (const idle of idleUsers) {
        for (const [challenger, challenge] of pendingChallenges) {
            if (challenge.target !== idle) continue;
            // |pm| only — see comment in /challenge handler for why we don't
            // also send |updatechallenges| (poke-env puts the challenger on
            // its `_challenge_queue` from BOTH frames, causing a double-/accept
            // and a stale "No pending challenge" log on the second attempt).
            sendToUser(idle, `|pm| ${challenger}| ${idle}|/challenge|${challenge.format}|||`);
            log(`Resent pending challenge ${challenger} -> ${idle} after battle cleanup`);
        }
    }
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
            // Real Showdown emits `>battle-tag\n|deinit` to confirm room
            // departure. Foul Play's `leave_battle` blocks on
            //    while True: msg = await recv(); if tag in msg and "deinit" in msg: return
            // so without this echo, FP hangs after every battle and never
            // loops to its next iter. Without this fix, even a 2-game run
            // against FP completes only the 1st battle.
            sendToUser(name, `>${tag}\n|deinit`);
            return;
        }
        if (cmd.startsWith('/choose ')) {
            // Real-Showdown clients (Foul Play) append "|<rqid>" for stale-move
            // detection: "/choose move swordsdance|42". BattleStream doesn't
            // interpret rqid; strip it before writing or the choice is rejected.
            const choice = cmd.slice(8).split('|')[0].trim();
            const slot = (name === entry.p1) ? 'p1' : 'p2';
            entry.players[slot].write(choice);
            return;
        }
        if (cmd.startsWith('/team ')) {
            // Team preview lead order: "/team 123456" or "/team 123456|<rqid>"
            const teamOrder = cmd.slice(1).split('|')[0].trim(); // "team 123456"
            const slot = (name === entry.p1) ? 'p1' : 'p2';
            entry.players[slot].write(teamOrder);
            return;
        }
        if (cmd.startsWith('/switch ')) {
            // Forced switch after a faint: Foul Play sends "/switch 2|<rqid>".
            // Real Showdown accepts /switch as a top-level command; BattleStream
            // expects "switch 2" written to the player stream (same as the
            // payload of /choose switch 2). Strip rqid, drop the leading '/'.
            const switchCmd = cmd.slice(1).split('|')[0].trim(); // "switch 2"
            const slot = (name === entry.p1) ? 'p1' : 'p2';
            entry.players[slot].write(switchCmd);
            return;
        }
        if (cmd.startsWith('/move ')) {
            // Symmetric with /switch — FP's format_decision can also emit
            // bare "/move <name>" in some paths. Strip rqid and forward.
            const moveCmd = cmd.slice(1).split('|')[0].trim(); // "move <name>"
            const slot = (name === entry.p1) ? 'p1' : 'p2';
            entry.players[slot].write(moveCmd);
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

        // Re-emit any |pm|/challenge that was issued to this user BEFORE they
        // logged in. With multiple concurrent V9RLPlayers (4+ wave parallelism),
        // the trainer races subprocess startup — challenges fire at iter start
        // before all FP/MM subprocesses have finished their /trn handshake.
        // Without this resend, those /pms are silently dropped by sendToUser
        // (no ws yet) and the corresponding RL player's send_challenges hangs
        // forever waiting for a battle that never starts. Same class as
        // cleanupBattle's pending-challenge resend (bug #7), but at login time.
        for (const [challenger, challenge] of pendingChallenges) {
            if (challenge.target !== id) continue;
            sendTo(ws, `|pm| ${challenger}| ${id}|/challenge|${challenge.format}|||`);
            log(`Resent pending challenge ${challenger} -> ${id} on login`);
        }
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

        // Send |pm|/challenge in Showdown standard format. Foul Play splits
        // this on `|` and strictly requires len == 9 with split[5] == format,
        // so the message has exactly 8 pipes (9 fields):
        //   ['', 'pm', ' sender', ' target', '/challenge', 'format', '', '', '']
        //
        // Why we DON'T also send |updatechallenges|: poke-env (in metamon's
        // 0.8.3.3 fork) registers the challenger on `_challenge_queue` from
        // BOTH `_update_challenges` (handles |updatechallenges|) AND
        // `_handle_challenge_request` (handles |pm|/challenge). Sending both
        // means two queue.put() calls for one challenge — `_accept_loop`'s
        // first iter consumes one, second iter consumes the duplicate and
        // immediately /accepts a stale challenger that's no longer pending.
        // Just sending |pm| keeps the queue at 1 entry per challenge. Foul
        // Play only reads |pm|, doesn't care about |updatechallenges|.
        sendToUser(target, `|pm| ${name}| ${target}|/challenge|${format}|||`);
        log(`Challenge: ${name} -> ${target} (${format})`);
        return;
    }

    if (cmdBody.startsWith('/accept ')) {
        const challenger = toId(cmdBody.slice(8).trim());
        const challenge = pendingChallenges.get(challenger);
        if (!challenge || challenge.target !== name) {
            log(`No pending challenge from ${challenger} for ${name} (raw msg=${msg.slice(0,80)})`);
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
        // Foul Play sends "/leave <battle-tag>" as a GLOBAL command (empty
        // room prefix), not a battle-room one — so this branch handles it
        // (the per-battle /leave at the top of handleMessage is unreachable
        // for FP). Echo back `>battle-tag\n|deinit` to unblock its
        // `leave_battle` await loop. Without this, FP hangs after every
        // single battle and never reaches its next iter.
        const tag = cmdBody.slice(7).trim();
        if (tag.startsWith('battle-')) {
            sendToUser(name, `>${tag}\n|deinit`);
        }
        return;
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
