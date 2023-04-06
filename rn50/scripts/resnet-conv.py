import time
t1 = time.time()
import torch
from torch.nn import Conv2d
import intel_extension_for_pytorch as ipex
import copy
import subprocess
import datetime
import os
import numpy as np
LOG_HOME=os.environ.get("LOG_HOME")
EMON_HOME=os.environ.get("EMON_HOME")
t2 = time.time()
a = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
b = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
def flush():
    #cmd = "bash /mnt/DP_disk2/resnet50/scripts/clear_cache.sh"
    #with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
    #    pass
    global a, b
    a += b

class EMON(object):
    def __init__(self, layer_id, flush):
        self.layer_id = str(layer_id)
        self.flush = "_flush" if flush else "_no-flush"
        self.log_home = LOG_HOME + '/layer' + self.layer_id + self.flush
        self.create_folder()

    def create_folder(self):
        os.system("mkdir " + self.log_home)

    def __enter__(self):
        self.emon_start()
    
    def __exit__(self, *args):
        self.emon_stop()
    
    
    def emon_start(self):
        cmd = "${EMON_HOME}/emon -v > " + self.log_home +"/emon-v.dat 2>&1 & ${EMON_HOME}//emon -M >  " + self.log_home +"/emon-M.dat 2>&1 & ${EMON_HOME}/emon -i /mnt/DP_disk2/resnet50/scripts/emon-config.txt > " + self.log_home + "/emon.dat 2>&1 &"
        #time.sleep(5)
        print("==========emon start")
        time.sleep(0.1)
        os.system(cmd)
        # subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def emon_stop(self):
        cmd = "${EMON_HOME}/emon -stop"
        time.sleep(0.1)
        print("==========emon stop")
        os.system(cmd)
        # subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)



class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        # layer1 Bottleneck0
        # self.convs = torch.nn.ModuleList()
        self.down_sampling = [4, 14, 27, 46]
        self.random_seed = 10
        np.random.seed(self.random_seed)
        print("set numpy random seed", self.random_seed)
        self.residual = [1, 11, 24, 43]
        self.keep_x = None
        convs = [
            Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            # layer1 bottleneck0
            Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            # layer1 bottleneck1
            Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            # layer1 bottleneck2
            Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            # layer2 bottleneck0
            Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False),
            # layer2 bottleneck1
            Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            # layer2 bottleneck2
            Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            # layer2 bottleneck3
            Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            # layer3 bottleneck0
            Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False),
            # layer3 bottleneck1
            Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            # layer3 bottleneck2
            Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            # layer3 bottleneck3
            Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            # layer3 bottleneck4
            Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            # layer3 bottleneck5
            Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            # layer4 bottleneck0
            Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False),
            # layer4 bottleneck1
            Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            # layer4 bottleneck2
            Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        ]
        self.convlist = torch.nn.ModuleList()
        for conv in convs:
            self.convlist.append(conv)
            

    def forward(self, x):
        c = 0
        for conv in self.convlist:
            if c in self.residual:
                self.keep_x = x
            if c in self.down_sampling:
                y = conv(self.keep_x)
            else:
                x = conv(x)
            c += 1
        return x
    
    # input in cache, weight not in cache, out ?
    def fw1(self, x, iters):
        for i in range(iters):
            self.forward(x)

    # input/output/weight in cache
    def fw2(self, x, iters):
        c = 0
        for conv in self.convlist:
            if c in self.residual:
                self.keep_x = x
            for i in range(iters):
                if c in self.down_sampling:
                    y = conv(self.keep_x)
                else:
                    z = conv(x)
                    if i == iters - 1:
                        x = z
            c += 1
        return x

    def prepare_inputs(self, x):
        self.conv_dict = {}
        c = 0
        for conv in self.convlist:
            self.conv_dict[conv] = {"layer_id": c}
            if c in self.residual:
                self.keep_x = x
            if c in self.down_sampling:
                self.conv_dict[conv]["input"] = self.keep_x
                y = conv(self.keep_x)
            else:
                self.conv_dict[conv]["input"] = x
                x = conv(x)
            c += 1

    def record_per_conv_time_fw2(self, iters):
        for conv in self.conv_dict:
            input = self.conv_dict[conv]['input']
            layer_id = self.conv_dict[conv]['layer_id']
            flush()
            with EMON(layer_id, flush=False):
                start = time.time()
                for _ in range(iters):
                    y = conv(input)
                end = time.time()
            ts_start = datetime.datetime.fromtimestamp(start).strftime('%m/%d/%Y %H:%M:%S.%f')[:-3]
            ts_end = datetime.datetime.fromtimestamp(end).strftime('%m/%d/%Y %H:%M:%S.%f')[:-3]            
            self.conv_dict[conv]['time_fw2'] = (end - start)
            t = self.conv_dict[conv]['time_fw2']
            print("for conv", conv, " layer_id =", layer_id, " duration", t, " start from", ts_start, " end at", ts_end)
            
    def record_per_conv_time_fw2_flush_weight(self, iters):
        for conv in self.conv_dict:
            same_conv = []
            for _ in range(iters):
                same_conv.append(copy.deepcopy(conv))
            input = self.conv_dict[conv]['input']
            layer_id = self.conv_dict[conv]['layer_id']
            random_idx = np.random.uniform(0, iters, iters)
            random_idx = [int(idx) for idx in random_idx]
            print("idx order for layer_id", layer_id, random_idx)
            flush()
            with EMON(layer_id, flush=True):
                start = time.time()
                for i in range(iters):
                    idx = random_idx[i]
                    y = same_conv[idx](input)
                end = time.time()
            ts_start = datetime.datetime.fromtimestamp(start).strftime('%m/%d/%Y %H:%M:%S.%f')[:-3]
            ts_end = datetime.datetime.fromtimestamp(end).strftime('%m/%d/%Y %H:%M:%S.%f')[:-3]
            del(same_conv)
            self.conv_dict[conv]['time_fw2_flush_weight'] = (end - start)
            t = self.conv_dict[conv]['time_fw2_flush_weight']
            print("for conv", conv, " layer_id =", layer_id, " duration", t, " start from", ts_start, " end at", ts_end)

    def display_gaps(self):
        total_t1 = 0
        total_t2 = 0
        for conv in self.conv_dict:
            t1 = self.conv_dict[conv]['time_fw2']
            t2  = self.conv_dict[conv]['time_fw2_flush_weight']
            self.conv_dict[conv]['gap'] = t2 - t1
            total_t1 += t1
            total_t2 += t2
        for conv in self.conv_dict:
            gap = self.conv_dict[conv]['gap']
            layer_id = self.conv_dict[conv]['layer_id']
            print("for conv", conv, " layer_id=", layer_id, " gap=", gap, " improvement ratio=", 100 * gap/total_t2, '%')

model = M().eval()
model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True, graph_mode=False, conv_bn_folding=False)

import argparse
parser = argparse.ArgumentParser(
    description="bench rn50"
)
parser.add_argument("--bench-approach", type=int, default=0)
parser.add_argument("--step", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()
x = torch.randn(args.batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last).bfloat16()

# t3=time.time()
# if args.bench_approach == 0:
#     model.fw1(x, args.step)
# else:
#     model.fw2(x, args.step)
model.prepare_inputs(x)
print("run_fw2 start from", datetime.datetime.fromtimestamp(time.time()).strftime('%m/%d/%Y %H:%M:%S.%f')[:-3])
model.record_per_conv_time_fw2(args.step)
print("run_fw2 end at", datetime.datetime.fromtimestamp(time.time()).strftime('%m/%d/%Y %H:%M:%S.%f')[:-3])
print("run_fw2_flush_weight start from", datetime.datetime.fromtimestamp(time.time()).strftime('%m/%d/%Y %H:%M:%S.%f')[:-3])
model.record_per_conv_time_fw2_flush_weight(args.step)
print("run_fw2_flush_weight end at", datetime.datetime.fromtimestamp(time.time()).strftime('%m/%d/%Y %H:%M:%S.%f')[:-3])
model.display_gaps()
# t4=time.time()

# print("total execute time: ", t4 - t1)
# print("module import and model init time: ", t3 - t1)
# print("benchmark time: ", t4 - t3)
# print("time/iter: ",  (t4 - t3) * 1000 /args.step, "ms")

