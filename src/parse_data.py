"""
16.412 Intent Inference GC | parse_data.py
Converts text representation of CARLA sim data into .json files for use with
Constant Velocity Method (CVM) trajectory prediction.
Author: Abbie Lee (abbielee@mit.edu)
"""

import json
import os
import argparse

class Agent:
    def __init__(self, fpath, labels = None):
        """
        fname: filepath to .txt for this agent
        labels: data labels from .txt
        """
        self.fpath = fpath
        self.fname = os.path.basename(self.fpath)

        self.ID = None
        self.set_id()

        self.labels = labels
        if self.labels == None: self.set_labels()

        # Maps frame number to (agent data dict, timestamp) tuple
        self.frames = {}

        self.process_file()

    def set_id(self):
        """
        Extracts id of agent from file name.
        """
        idx = self.fname.find(".txt")
        self.ID = int(''.join([c for c in self.fname[:idx] if c.isdigit()]))

    def set_labels(self):
        f = open(self.fpath, mode='r')
        line = f.readline().rstrip()
        self.labels = line.split(" | ")

    def __equal__(self, other):
        if self.ID == other.ID:
            return True
        return False

    def __repr__(self):
        return "Agent ID: " + str(self.ID)

    def process_file(self):
        """
        Currently only works with this label structure:

        ["frame", "timestamp", "position", "velocity", "heading", \
                  "angular velocity", "acceleration", "light status"]

        Should make more generalizable later. If possible? Might not be.
        """
        with open(self.fpath, mode='r') as f:
            next(f) # skip header row
            for line in f:
                elements = line.split(" | ")

                frame = int(elements[0])
                timestamp = round(float(elements[1]), 2)

                pos = [round(float(e), 2) for e in elements[2].split(',')]
                vel = [round(float(e), 2) for e in elements[3].split(',')]
                heading = float(elements[4])
                angvel = [round(float(e), 2) for e in elements[5].split(',')]
                accel = [round(float(e), 2) for e in elements[6].split(',')]

                light = [int(e) for e in elements[7].split(',')]

                # 2D world representation
                agent_dict = {}
                agent_dict["position"] = pos[:2]
                agent_dict["velocity"] = vel[:2]
                agent_dict["heading"] = heading
                agent_dict["angular velocity"] = angvel[2] # only z component (yaw rate)
                agent_dict["acceleration"] = accel[:2]
                agent_dict["light status"] = {"stopped": light[0], "lightID": light[1]}
                agent_dict["id"] = self.ID

                # Add data to frame
                self.frames[frame] = (agent_dict, timestamp)

    def to_dict(self, frameID):
        """
        Returns agent state dict for a given frame ID.
        """
        return self.frames[frameID][0]

class Frame:
    def __init__(self, frameID, timestamp):
        self.frameID = frameID
        self.agents = {} # active agents in this frame; maps agentID to Agent object
        self.timestamp = timestamp
        self.size = 0

    def add_agent(self, agent):
        if agent.ID not in self.agents.keys():
            self.agents[agent.ID] = agent
            self.size += 1

    def __equal__(self, other):
        if self.frameID == other.frameID:
            return True
        return False

    def __repr__(self,):
        return "Frame ID: " + str(self.frameID) + ", Timestamp: " + str(self.timestamp)

    def to_dict(self):
        agents = [a.to_dict(self.frameID) for a in self.agents.values()]
        frame_dict = {"timestamp": self.timestamp, "object_list": agents, \
                      "frame": self.frameID, "size": self.size}
        return frame_dict

class MultiAgentScene:
    def __init__(self, dirpath, labels = None):
        self.dirpath = dirpath
        self.labels = labels
        self.agents = [] # all the agents in this scene
        self.frames = {} # maps frame ID to Frame objects
        self.size = len(self.frames)
        self.gen_scene()

    def gen_scene(self):
        """
        Iterates over .txt files and populations MultiAgentScene with relevant
        Frames, and populates Frames with relevant Agents.
        """
        directory = os.fsencode(self.dirpath)

        # Create agents
        for file in os.listdir(directory):
             filename = os.fsdecode(file)
             if filename.endswith(".txt"):
                 new_agent = Agent(self.dirpath+filename, self.labels)
                 self.agents.append(new_agent)

        # create frames
        for ag in self.agents:
            for fr, data in ag.frames.items():
                ag_dict = data[0]
                timestamp = data[1]
                if fr in self.frames.keys():
                    self.frames[fr].add_agent(ag)
                else:
                    new_frame = Frame(fr, timestamp)
                    new_frame.add_agent(ag)
                    self.frames[fr] = new_frame

        self.size = len(self.frames)

    def to_json_dataset(self, outpath):
        """
        Writes .json files for this scene to the specified outpath.
        """
        counter = 1
        frames = sorted(self.frames.values(), key = lambda f : f.frameID)
        for fr in frames:
            with open(outpath + "data/"+ str(counter) + ".json", mode="w") as json_file:
                json.dump(fr.to_dict(), json_file)
            counter += 1

        # create dataset info file
        ds_name = os.path.basename(outpath[:-1])
        with open(outpath + "dataset_info.json", mode="w") as json_file:
            json.dump({"dataset_name": ds_name}, json_file)

def parse_commandline():
    parser = argparse.ArgumentParser(description='Parses .txt from CARLA to .json for use with CVM-based predictors.')
    parser.add_argument('--data', required=True, action='store', help='Path to .txt data files')
    parser.add_argument('--outpath', required=True, action='store', help='Path to directory for output files')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_commandline()
    data_path = args.data
    out_path = args.outpath

    if out_path[-1] != "/":
        out_path += "/"

    labels = ["frame", "timestamp", "position", "velocity", "heading", \
              "angular velocity", "acceleration", "light status"]

    MAS = MultiAgentScene(data_path, labels)
    MAS.to_json_dataset(out_path)
