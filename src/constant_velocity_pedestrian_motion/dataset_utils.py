import json


class DetectedObject:

    def __init__(self, obj_id, position, angvel=0):
        self.id = obj_id
        self.position = position
        self.angvel = angvel

    def from_json(obj_json):
        if 'angular velocity' in obj_json.keys():
            return DetectedObject(obj_json['id'], obj_json['position'], obj_json['angular velocity'])
        return DetectedObject(obj_json['id'], obj_json['position'])

class Detection:

    def __init__(self, timestamp, size, object_list, detectionID):
        self.timestamp = timestamp
        self.size = size
        self.object_list = object_list
        self.detectionID = detectionID

    def from_json(detection_json, ID = None):
        timestamp = detection_json['timestamp']
        size = detection_json['size']
        object_list = []
        for obj_json in detection_json['object_list']:
            object_list.append(DetectedObject.from_json(obj_json))
        return Detection(timestamp, size, object_list, ID)

    def objects(self):
        return self.object_list

    def __len__(self):
        return len(self.object_list)

    def __repr__(self):
        return "DETECTION OBJECT" + "\n" + \
               "ID: " + str(self.detectionID) + "\n" + \
               "Timestamp: " + str(self.timestamp) + "\n" + \
               "Size: " + str(self.size) + "\n" + \
               "----------------"

class Trajectory():

    def __init__(self, obj_id, start_time, positions=None, angvels=None):
        self.obj_id = obj_id
        self.start_time = start_time
        self.positions = [] if positions is None else positions
        self.angvels = [] if angvels is None else angvels

    def add_position(self, position):
        self.positions.append(position)

    def add_angvel(self, angvel):
        self.angvels.append(angvel)

    def __len__(self):
        return len(self.positions)

    def __repr__(self):
        return "TRAJECTORY OBJECT" + "\n" + \
               "Agent ID: " + str(self.obj_id) + "\n" + \
               "Start Time: " + str(self.start_time) + "\n" + \
               "Positions: " + str(self.positions) + "\n" + \
               "Angvels: " + str(self.angvels) + "\n" + \
               "Num Positions: " + str(len(self.positions)) + "\n" + \
               "----------------"

class Sample():
    def __init__(self, obj_id, start_time, positions=None, angvels=None, detectionID=None):
        self.trajectory = Trajectory(obj_id, start_time, positions=positions, angvels=angvels)
        self.detectionID = detectionID # ID of dectection frame that this sample is in

    @classmethod
    def from_trajectory(cls, trajectory, detectionID=None):
        return cls(trajectory.obj_id, trajectory.start_time, \
                   positions=trajectory.positions, angvels=trajectory.angvels, \
                   detectionID=detectionID)

    def add_position(self, position):
        self.trajectory.add_position(position)

    def add_angvel(self, angvel):
        self.trajectory.add_angvel(angvel)

    @property
    def positions(self):
        return self.trajectory.positions

    @property
    def angvels(self):
        return self.trajectory.angvels

    @property
    def obj_id(self):
        return self.trajectory.obj_id

    def __len__(self):
        return len(self.trajectory)

    def __repr__(self):
        return "SAMPLE OBJECT" + "\n" + \
               "Agent ID: " + str(self.trajectory.obj_id) + "\n" + \
               "Detection ID: " + str(self.detectionID) + "\n" + \
               "Trajectory: " + str(self.trajectory) + "\n" + \
               "----------------"

    def slice(self, sequence_length, min_length):
        """
        Slice the sample into multiple samples of at least length min_length and maximum sequence_length with
        a sliding window and step size 1.
        """

        if len(self) < min_length:
            return []

        split_samples = []
        start_index = 0
        end_index = sequence_length
        while start_index < len(self):
            new_trajectory = Trajectory(self.obj_id, self.trajectory.start_time + start_index, \
                                        positions=self.positions[start_index:end_index], \
                                        angvels=self.angvels[start_index:end_index])
            new_sample = Sample.from_trajectory(new_trajectory, detectionID=self.detectionID)
            start_index += 1
            end_index = start_index + sequence_length

            if len(new_sample) < min_length: # don't consider too short samples
                continue

            split_samples.append(new_sample)

        return split_samples
