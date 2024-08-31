from sklearn.preprocessing import LabelEncoder # preprocessing into numeric
from sklearn.svm import SVC
import pickle

#initializing  of embedding & recognizer 
embeddingFile="output/embedding.pickle" # initialize the file
#new & empty file
recognizerFile="output/recognizer.pickle" 
labelEncFile = "output/labelEncoder.pickle" 

# loading the embedding
print("Loading the embedding")
data = pickle.loads(open(embeddingFile, "rb").read())

print("Loading the label encoder")
labelEncoder = LabelEncoder()
labels=labelEncoder.fit_transform(data["names"])

print("Training model...")
recognizer = SVC(C=1.0,kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f= open(recognizerFile, "wb")
f.write(pickle.dumps(recognizer))
f.close()

f= open(labelEncFile, "wb")
f.write(pickle.dumps(labelEncoder))
f.close()

