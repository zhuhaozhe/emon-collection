export LD_PRELOAD="${CONDA_PREFIX}/lib/libjemalloc.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto"
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:$LD_PRELOAD

seq=`date +%m%d%H%M%S`
export EMON_HOME=/home/sparkuser/sep_private_5_33_linux_0316081130eb678/bin64
source /home/sparkuser/sep_private_5_33_linux_0316081130eb678/sep_vars.sh

function run(){
  if [ $1 = "on" ]
  then
    emon --write-msr 0x1a4=0x20
    echo "Run test for Prefetch ON"
    LOG_HOME_BASE=/mnt/DP_disk2/resnet50/results/${seq}-find-important-conv-prefetch-on
  fi
  if [ $1 = "off" ]
  then
    emon --write-msr 0x1a4=0x2f
    echo "Run test for Prefetch OFF"
    LOG_HOME_BASE=/mnt/DP_disk2/resnet50/results/${seq}-find-important-conv-prefetch-off
  fi
  mkdir $LOG_HOME_BASE
  cat /proc/cpuinfo | cat > ${LOG_HOME_BASE}/cpu.info
  cat /proc/meminfo | cat > ${LOG_HOME_BASE}/mem.info
  cp /mnt/DP_disk2/resnet50/scripts/rn50.sh ${LOG_HOME_BASE}/rn50.sh
  cp /mnt/DP_disk2/resnet50/scripts/resnet-conv.py ${LOG_HOME_BASE}/resnet-conv.py
  emon -v |grep Prefetching | cat > ${LOG_HOME_BASE}/prefetching.stat
  
  uname -a | cat > ${LOG_HOME_BASE}/kernel.info
  
  for REPEAT in {1..2}
  do
    LOG_HOME_REPEAT=$LOG_HOME_BASE/repeat-${REPEAT}
    if [[ -a $LOG_HOME_REPEAT ]];then
      rm -rf $LOG_HOME_REPEAT
    fi
    mkdir $LOG_HOME_REPEAT
    echo "Repeat ${REPEAT}, write to ${LOG_HOME_REPEAT}"
        STEP=8000
	for BS in 16 4 1
	do
          export LOG_HOME=$LOG_HOME_REPEAT/iters_${STEP}_BS_${BS}
          if [[ -a $LOG_HOME ]];then
            rm -rf $LOG_HOME
          fi;
          mkdir $LOG_HOME
          echo "fw mode with step ${STEP}, BS ${BS}, write to ${LOG_HOME}"
          #$EMON_HOME/emon -v > ${LOG_HOME}/emon-v.dat 2>&1 &
          #$EMON_HOME/emon -M > ${LOG_HOME}/emon-M.dat 2>&1 &
          #$EMON_HOME/emon -i /mnt/DP_disk2/resnet50/scripts/emon-config.txt > $LOG_HOME/emon.dat 2>&1 &
          numactl -C 0-31 -m 0 python -u ./scripts/resnet-conv.py --batch-size=$BS --step=$STEP 2>&1 |tee $LOG_HOME/app-level-info.log
          #$EMON_HOME/emon -stop
          sudo pkill python
          sleep 1
          bash scripts/clear_cache.sh
	done
  done
}

run "on"
run "off"

