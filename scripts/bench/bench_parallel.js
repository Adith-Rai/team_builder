// Parallel BattleStream — N concurrent battles in a single Node process.
// Tests whether one Node process can pool many battles efficiently.
// Usage: node bench_parallel.js <format> <concurrency> <total>
const { BattleStream, getPlayerStreams } = require('pokemon-showdown/dist/sim');

async function runBattle(format, id) {
  const stream = new BattleStream({ keepAlive: false });
  const players = getPlayerStreams(stream);
  stream.write('>start ' + JSON.stringify({ formatid: format }));
  stream.write('>player p1 ' + JSON.stringify({ name: 'P1', team: '' }));
  stream.write('>player p2 ' + JSON.stringify({ name: 'P2', team: '' }));

  let omniscientTurn = 0;
  let ended = false;

  const drain = async (slot, ps) => {
    let chunk;
    while ((chunk = await ps.read()) !== null) {
      if (ended) return;
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (line.startsWith('|win|') || line === '|tie' || line.startsWith('|tie|')) {
          ended = true;
          return;
        }
        if (line.indexOf('|turn|') === 0 && slot === 'p1') {
          omniscientTurn = parseInt(line.slice(6)) || omniscientTurn;
        } else if (line.indexOf('|request|') === 0) {
          let req;
          try { req = JSON.parse(line.slice('|request|'.length)); } catch (e) { continue; }
          if (req.wait) continue;
          if (req.teamPreview) stream.write('>' + slot + ' team 123456');
          else stream.write('>' + slot + ' default');
        }
      }
    }
  };

  await Promise.all([drain('p1', players.p1), drain('p2', players.p2)]);
  return { id, turns: omniscientTurn };
}

async function main() {
  const fmt = process.argv[2] || 'gen9randombattle';
  const CONCURRENCY = parseInt(process.argv[3] || '32');
  const TOTAL = parseInt(process.argv[4] || '128');
  const tStart = Date.now();
  let nextId = 0;
  const inFlight = new Set();
  const results = [];

  const launch = () => {
    if (nextId >= TOTAL) return null;
    const id = nextId++;
    const p = runBattle(fmt, id).then(r => {
      results.push(r);
      inFlight.delete(p);
    });
    inFlight.add(p);
    return p;
  };

  for (let i = 0; i < CONCURRENCY && nextId < TOTAL; i++) launch();
  while (inFlight.size > 0) {
    await Promise.race(inFlight);
    while (inFlight.size < CONCURRENCY && nextId < TOTAL) launch();
  }

  const tEnd = Date.now();
  const totalTurns = results.reduce((s, r) => s + r.turns, 0);
  const totalMs = tEnd - tStart;
  console.log('Parallel BattleStream, conc=' + CONCURRENCY + ', total=' + TOTAL + ' battles of ' + fmt + ':');
  console.log('  total turns:    ' + totalTurns);
  console.log('  total wall:     ' + totalMs + 'ms');
  console.log('  turns/sec:      ' + (totalTurns / (totalMs / 1000)).toFixed(1));
  console.log('  battles/sec:    ' + (TOTAL / (totalMs / 1000)).toFixed(2));
  console.log('  avg turns/btl:  ' + (totalTurns / TOTAL).toFixed(1));
  console.log('  avg ms/battle:  ' + (totalMs / TOTAL).toFixed(0));
}
main().catch(e => { console.error(e); process.exit(1); });
