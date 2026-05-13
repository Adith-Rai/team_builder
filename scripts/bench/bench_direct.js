// Direct BattleStream benchmark — no WS, no poke-env. Pure sim throughput.
// Usage: node bench_direct.js <format> <n_battles>
//   format = gen9randombattle | gen9ou | gen8randombattle | ...
//   n_battles = how many sequential battles to run
const { BattleStream, getPlayerStreams } = require('pokemon-showdown/dist/sim');

async function runBattle(format) {
  const stream = new BattleStream({ keepAlive: false });
  const players = getPlayerStreams(stream);
  stream.write('>start ' + JSON.stringify({ formatid: format }));
  stream.write('>player p1 ' + JSON.stringify({ name: 'P1', team: '' }));
  stream.write('>player p2 ' + JSON.stringify({ name: 'P2', team: '' }));

  let turnCount = 0;
  let omniscientTurn = 0;
  let p1Choices = 0;
  let p2Choices = 0;
  let winner = null;
  let ended = false;

  const drain = async (slot, ps) => {
    let chunk;
    while ((chunk = await ps.read()) !== null) {
      if (ended) return;
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (line.startsWith('|win|')) {
          ended = true;
          winner = line.slice(5).trim();
          return;
        }
        if (line === '|tie' || line.startsWith('|tie|')) {
          ended = true;
          winner = 'tie';
          return;
        }
        if (line.indexOf('|turn|') === 0) {
          if (slot === 'p1') omniscientTurn = parseInt(line.slice(6)) || omniscientTurn;
        } else if (line.indexOf('|request|') === 0) {
          let req;
          try { req = JSON.parse(line.slice('|request|'.length)); } catch (e) { continue; }
          if (req.wait) continue;
          if (req.teamPreview) {
            stream.write('>' + slot + ' team 123456');
          } else {
            stream.write('>' + slot + ' default');
          }
          if (slot === 'p1') p1Choices++;
          else p2Choices++;
        }
      }
    }
  };

  const t0 = process.hrtime.bigint();
  await Promise.all([drain('p1', players.p1), drain('p2', players.p2)]);
  const elapsedMs = Number(process.hrtime.bigint() - t0) / 1e6;
  return { turns: omniscientTurn, p1Choices, p2Choices, elapsedMs, winner };
}

async function main() {
  const fmt = process.argv[2] || 'gen9randombattle';
  const N = parseInt(process.argv[3] || '50');
  const tStart = Date.now();
  const results = [];
  for (let i = 0; i < N; i++) {
    const r = await runBattle(fmt);
    results.push(r);
  }
  const tEnd = Date.now();
  const totalTurns = results.reduce((s, r) => s + r.turns, 0);
  const totalChoices = results.reduce((s, r) => s + r.p1Choices + r.p2Choices, 0);
  const totalMs = tEnd - tStart;
  console.log('Direct BattleStream, ' + N + ' battles of ' + fmt + ':');
  console.log('  total turns:   ' + totalTurns);
  console.log('  total choices: ' + totalChoices + ' (across both sides)');
  console.log('  total wall:    ' + totalMs + 'ms');
  console.log('  turns/sec:     ' + (totalTurns / (totalMs / 1000)).toFixed(1));
  console.log('  choices/sec:   ' + (totalChoices / (totalMs / 1000)).toFixed(1));
  console.log('  battles/sec:   ' + (N / (totalMs / 1000)).toFixed(2));
  console.log('  avg turns/btl: ' + (totalTurns / N).toFixed(1));
  console.log('  avg ms/battle: ' + (totalMs / N).toFixed(0));
  console.log('  winners: ' + JSON.stringify(results.reduce((m, r) => { m[r.winner || 'none'] = (m[r.winner || 'none'] || 0) + 1; return m; }, {})));
}
main().catch(e => { console.error(e); process.exit(1); });
