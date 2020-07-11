import glob
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.utils import np_utils

from music21 import converter, instrument, note, stream, chord

# creating an empty list to hold the notes in
notes = []

# this for loop goes through each midi file and flattens out the notes inside of it
for file in glob.glob("lofi-samples/samples/*.mid"):
    midi = converter.parse(file)
    notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element,
                      note.Note):  # if it's a single note, we don't have to join it to any other notes in the series
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):  # if it's a chord, we will have to join it to the other notes
            notes.append('.'.join(str(n) for n in element.normalOrder))

# this is the amount of previous notes our algorithm will use to predict the next notes
# mess around with this number to see how this impacts the accuracy
sequence_length = 20  # our chord progressions are pretty short so we might not need that many notes

# get all pitch names
pitchnames = sorted(set(item for item in notes))

# create a dictionary to map pitches to integers
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

network_input = []
network_output = []
n_patterns = 0

# create input sequences and the corresponding outputs
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)

# reshape the input vector into a format compatible with LSTM layers
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
# normalizing the input values
n_vocab = len(set(notes))
network_input = network_input / float(n_vocab)

try:
    network_output = np_utils.to_categorical(network_output)
except ValueError:  # raised if `y` is empty.
    print("Value Error empty 'y'")

model = Sequential()

# each model.add command adds a new layer to our sequential model
# this one is our input layer :)
model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))

model.add(Dropout(0.3))
# these layers will set a fraction of inputs (in in this case 2/10) to 0 at each update.
# it's a technique to prevent overfitting
# (in case you haven't heard, the fraction of input units we're dropping during training is our first parameter)

model.add(LSTM(512, return_sequences=True))
# each type of LSTM layer takes a sequence as an input and returns either sequences or matrixes
# here, the first parameter is how many nodes our layer will have.
# (same thing with all the non-dropout layers)

model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
# these guys are fully connected and attach to an output node

model.add(Dense(n_vocab))
# because this one's our last layer,
# it should have the same amount of nodes as the number of different outputs our system has
# this will make sure the network's output will map right onto the system classes

model.add(Activation('softmax'))
# this one figures out which activation function to use to calculate the output

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
# this is our training command
# we're using categorical

model.fit(network_input, network_output, epochs=20, batch_size=64)

start = np.random.randint(0, len(network_input) - 1)
int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

pattern = network_input[start]

prediction_output = []

for note_index in range(100):  # here, we're generating 100 notes
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)

    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)

    result = int_to_note[int(index)]
    prediction_output.append(result)

    pattern.ravel()
    # beta = pattern + index
    # beta = beta[1:len(beta)]

offset = 0
output_notes = []

for pattern in prediction_output:
    # Individual Chords
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []  # Creating the array where we'll store the note values, which the for loop below will handle
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)  # Adding the note to the chord object
        new_chord.offset = offset  # Connecting it to the offset variable
        output_notes.append(new_chord)  # Adding it to the song
    # Notes created
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
    # Make sure notes don't end up on top of each other by adding an 0.5 offset every time
    offset += 0.5

s = stream.Stream(output_notes)
mf = s.write('midi', fp="lofi-samples/testOutput.mid")
