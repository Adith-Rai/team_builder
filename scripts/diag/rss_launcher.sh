#!/bin/bash
pkill -f rss_watcher.sh 2>/dev/null
sleep 1
nohup /workspace/scripts/rss_watcher.sh > /dev/null 2>&1 < /dev/null &
echo "launched pid $!"
sleep 2
ps -ef | grep rss_watcher | grep -v grep
echo "--- log ---"
ls -la /workspace/logs/rss_watcher.log 2>/dev/null
cat /workspace/logs/rss_watcher.log 2>/dev/null
