import os 
import librosa
import math
import json

DATASET_PATH = "genres_original"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050 #number of smaples per second taken from an analog signal to make it digital
DURATION = 30 #in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# hop_length: The number of samples between successive frames -> the columns of a spectrogram.

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    #dictionary to store data
    data = {
        "mapping": [], #genres
        "mfcc": [], #training data
        "labels": [] #outputs
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    #loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        #ensure we are not at the root (dataset) level
        if dirpath is not dataset_path:

            #save the semantic label
            dirpath_components = dirpath.split("/") #genre/blues = ["genre", "blues"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            #process files for a specific genre
            for f in filenames:

                #load audio file
                file_path = os.path.join(dirpath,f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s #s=0 -> 0
                    finish_sample = num_samples_per_segment + start_sample # s=0 -> num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft = n_fft,
                                                n_mfcc = n_mfcc,
                                                hop_length = hop_length)
                    
                    mfcc = mfcc.T

                    
                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)