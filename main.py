from htm.bindings.algorithms import TemporalMemory
from htm.bindings.sdr import SDR
from htm.encoders.rdse import RDSE, RDSE_Parameters
import argparse
import itertools
import numpy as np
import random
import time

default_parameters = {
    'num_cells': 2000,
    'local_sparsity': .02,
    'apical_denrites': {
        'activationThreshold': 20,
        'minThreshold': 14,
        'maxNewSynapseCount': 32,
        'connectedPermanence': 0.5,
        'initialPermanence': 0.21,
        'permanenceIncrement': 0.05,
        'permanenceDecrement': 0.025,
        'predictedSegmentDecrement': 0.002,
        'maxSynapsesPerSegment': 40,
        'maxSegmentsPerCell': 40,
    },
}

conscious_threshold = 20/100

BACKGROUND = " ."

class World:
    def __init__(self, dims, objects):
        # Initialize game board.
        self.dims = tuple(dims)
        self.coordinates = list(itertools.product(*(range(x) for x in self.dims)))
        # Put the objects at random locations.
        locations = random.sample(self.coordinates, len(objects))
        self.objects = {p: locations.pop() for p in objects}

    def state(self):
        """ Returns 2D grid of the objects and the BACKGROUND """
        data = np.full(self.dims, BACKGROUND, dtype=object)
        for character, location in self.objects.items():
            data[location] = character
        return data

    def draw(self):
        data = self.state()
        string = ""
        for row in range(self.dims[0]):
            for col in range(self.dims[1]):
                string += data[row, col]
            string += "\n"
        return string[:-1]

    def draw_colored(self, colors):
        data = self.state()
        string = ""
        for row in range(self.dims[0]):
            for col in range(self.dims[1]):
                character = data[row, col]
                color = colors[row, col]
                if not bool(color):
                    string += character
                elif color == "red":
                    string += '\033[1m\033[41m' + character + '\033[0m'
            string += "\n"
        return string[:-1]

    def advance(self):
        # Make the players walk in random directions.
        for character, (row, col) in self.objects.items():
            destinations = []
            for (offset_row, offset_col) in [
                    [-1, -1],
                    [-1,  0],
                    [-1, +1],
                    [ 0, -1],
                    [ 0,  0],
                    [ 0, +1],
                    [+1, -1],
                    [+1,  0],
                    [+1, +1],
                ]:
                next_row = row + offset_row
                next_col = col + offset_col
                if next_row not in range(self.dims[0]): continue
                if next_col not in range(self.dims[1]): continue
                if self.state()[next_row, next_col] not in (character, BACKGROUND): continue
                destinations.append((next_row, next_col))
            self.objects[character] = random.choice(destinations)

class Model:
    def __init__(self, world, parameters=default_parameters):
        self.world = world
        self.area_size = parameters['num_cells']
        self.num_areas = len(self.world.coordinates)
        # Make an RDSE for every location.
        self.enc = np.zeros(self.world.dims, dtype=object)
        enc_parameters = RDSE_Parameters()
        enc_parameters.size = self.area_size
        enc_parameters.sparsity = parameters['local_sparsity']
        enc_parameters.category = True
        for coords in self.world.coordinates:
            self.enc[coords] = RDSE(enc_parameters)
        # Make empty buffers for the working data.
        self.local = np.zeros(self.world.dims, dtype=object)
        self.gnw   = np.zeros(self.world.dims, dtype=object)
        for coords in self.world.coordinates:
            self.local[coords] = SDR((self.area_size,))
            self.gnw[coords]   = SDR((self.area_size,))
        # Make an instance of the model at every location.
        self.apical_denrites = np.zeros(self.world.dims, dtype=object)
        self.gnw_size = self.num_areas * self.area_size
        for coords in self.world.coordinates:
            self.apical_denrites[coords] = TemporalMemory(
                    [self.area_size], # column_dimensions
                    cellsPerColumn = 1,
                    externalPredictiveInputs = self.gnw_size,
                    seed = 0,
                    **parameters['apical_denrites'])

    def reset_attention(self):
        for coords in self.world.coordinates:
            self.gnw[coords] = SDR((self.area_size,))

    def advance(self, learn=True):
        self.world.advance()
        world_data = self.world.state()
        # Compute the local activity by encoding the sensory data into an SDR.
        self.local = np.zeros(self.world.dims, dtype=object)
        for idx, coords in enumerate(self.world.coordinates):
            character = world_data[coords]
            enc = self.enc[coords]
            if character == BACKGROUND:
                self.local[coords] = SDR((self.area_size,))
            else:
                self.local[coords] = enc.encode(ord(character))
        # Compute the apical dendrites.
        prev_gnw = SDR((self.gnw_size,)).concatenate(list(self.gnw.flat))
        self.gnw = np.zeros(self.world.dims, dtype=object)
        for coords in self.world.coordinates:
            self.apical_denrites[coords].reset()
            self.apical_denrites[coords].activateDendrites(True, prev_gnw, prev_gnw)
            apical_activity = self.apical_denrites[coords].getPredictiveCells().reshape((self.area_size,))
            self.gnw[coords] = SDR((self.area_size,)).intersection(self.local[coords], apical_activity)
            self.apical_denrites[coords].activateCells(self.local[coords], True)

    def attention_map(self):
        percent = np.empty(self.world.dims)
        for coords in self.world.coordinates:
            gnw_sparsity = self.gnw[coords].getSparsity()
            local_sparsity = self.local[coords].getSparsity()
            if local_sparsity == 0:
                percent[coords] = 0.;
            else:
                percent[coords] = gnw_sparsity / local_sparsity
        return percent

    def draw_heatmap(self):
        attention = np.zeros(self.world.dims, dtype=object)
        attention[self.attention_map() >= conscious_threshold] = "red"
        return self.world.draw_colored(attention)

    def promote_region(self, coordinates=None, verbose=False):
        if coordinates is None:
            coordinates = random.choice(self.world.coordinates)
        self.gnw[coordinates] = self.local[coordinates]
        if verbose: print("Promote (%d, %d) to attention."%coordinates)

    def promote_object(self, character=None, verbose=False):
        if character is None:
            character, location = random.choice(list(self.world.objects.items()))
        else:
            location = self.world.objects[character]
        self.promote_region(location)
        if verbose: print("Promote %s to attention."%character)
        return character

    def run(self, iterations, train_not_test, character=None, verbose=True):
        attention_spans = [];
        self.reset_attention()
        for t in range(iterations):
            message = ""
            if not any(x >= conscious_threshold for x in self.attention_map().flat):
                current_episode_length = 0
                obj = self.promote_object(character)
                # self.promote_region(verbose=verbose)
                message += "Promoted %s to \nconscious attention.\n"%obj
            self.advance()
            if verbose:
                heatmap = self.draw_heatmap()
            attn = self.attention_map()
            if train_not_test:
                if any(x >= conscious_threshold for x in self.attention_map().flat):
                    current_episode_length += 1
                else:
                    attention_spans.append(current_episode_length)
                    current_episode_length = None
            else:
                if attn[self.world.objects[obj]] >= conscious_threshold and (
                        sum(attn.flat >= conscious_threshold) == 1):
                    current_episode_length += 1
                else:
                    message += "Attention failed.\n"
                    attention_spans.append(current_episode_length)
                    current_episode_length = None
                    self.reset_attention()
            if verbose:
                if not train_not_test: print("\033c", end='')
                print("=="*self.world.dims[1])
                print(heatmap)
                print("=="*self.world.dims[1])
                while message.count("\n") < 3: message += "\n"
                print(message, end='')
                print("=="*self.world.dims[1])
                print("\n")
                if train_not_test:      pass
                elif message.strip():   time.sleep(1)
                else:                   time.sleep(1/7)
        if current_episode_length is not None: attention_spans.append(current_episode_length)
        avg_attention_span = np.mean(attention_spans)
        return avg_attention_span

def main(parameters, argv=None, verbose=False):
    parser = argparse.ArgumentParser()
    args = parser.parse_args(argv)

    w = World([6,10], "🐱🐶🏀🦍")
    m = Model(w)
    session_length = 20
    num_actions = w.dims[0]*w.dims[1]*len(w.objects) * 10
    for epoch in range(num_actions):
        m.run(session_length, train_not_test=True, verbose=False)
        if verbose and epoch % 2 == 0:
            print("\033cTraining %d%%"%int(100 * epoch / num_actions))
    if verbose: m.run(100, train_not_test=False, verbose=verbose)
    score = m.run(2000, train_not_test=False, verbose=False)
    if verbose: print(str(m.apical_denrites[0,0].connections))
    if verbose: print("Average attention span:", score)
    return score

if __name__ == "__main__": main(default_parameters, verbose=True)
