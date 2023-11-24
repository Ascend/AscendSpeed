ps -ef | grep generate | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep benchmark | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep run | grep -v grep | awk '{print $2}' | xargs kill -9
pkill -9 python