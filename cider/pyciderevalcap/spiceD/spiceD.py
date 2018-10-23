from __future__ import division
import os
# import ipdb
import sys
import os
import subprocess
import threading
import json
import numpy as np
import ast
import tempfile
import urllib.request
import time


# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_JAR = 'spice-1.0.jar'
TEMP_DIR = 'tmp'
CACHE_DIR = 'cache'
# IP_PORT = 'http://100.88.64.67:7777'


class SpiceD:
    def float_convert(self, obj):
        try:
            return float(obj)
        except:
            return np.nan

    def compute_score(self, gts, res, ip_str, port_str):
        assert(sorted(gts.keys()) == sorted(res.keys()))
        imgIds = sorted(gts.keys())
        
        # Prepare temp input file for the SPICE scorer
        input_data = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            input_data.append({"image_id": id, "test": hypo[0], "refs": ref})

        time_1 = time.time()
        # pid = os.getpid()
        cwd = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(cwd, TEMP_DIR + '_' + port_str)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        in_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, dir=temp_dir, encoding='utf-8')
        json.dump(input_data, in_file, indent=2)
        in_file.close()

        # Start job
        out_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, dir=temp_dir, encoding='utf-8')
        out_file.close()
        cache_dir = os.path.join(cwd, CACHE_DIR + '_' + port_str)  # CACHE_DIR_port
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        time_2 = time.time()
        IP_PORT = 'http://' + ip_str + ':' + port_str
        g_cgi = IP_PORT + '/test?in=' + \
                in_file.name + '&out=' + \
                out_file.name + '&cache=' + \
                cache_dir + '&subset=1&silent=1'
        req = urllib.request.Request(url=g_cgi)
        time_3 = time.time()
        res = urllib.request.urlopen(req).read()
        time_4 = time.time()
        json_res = json.loads(res)
        print("spice-1: {:.3f}".format((time_3 - time_2)))
        print("spice-2: {:.3f}".format((time_4 - time_3)))

        # time_3 = time.time()
        # spice_cmd = ['java', '-jar', '-Xmx8G', SPICE_JAR, in_file.name,
        #              '-cache', cache_dir,
        #              '-out', out_file.name,
        #              '-subset',
        #              '-silent']
        #
        # subprocess.check_call(spice_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

        # Read and process results
        with open(out_file.name) as data_file:    
            results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)

        imgId_to_scores = {}
        spice_scores = []
        for item in results:
            imgId_to_scores[item['image_id']] = item['scores']
            spice_scores.append(self.float_convert(item['scores']['All']['f']))
        average_score = np.mean(np.array(spice_scores))
        
        scores = []
        for image_id in imgIds:
            score_set = {}
            for category,score_tuple in imgId_to_scores[image_id].items():
                score_set[category] = {k: self.float_convert(v) for k, v in score_tuple.items()}
            scores.append(score_set)
        time_4 = time.time()

        # print("file time: {:.3f}".format((time_2 - time_1)))
        # print("call time: {:.3f}".format((time_3 - time_2)))
        # print("stat time: {:.3f}".format((time_4 - time_3)))

        #return average_score, scores
        return average_score, spice_scores

    def method(self):
        return "SPICE"


