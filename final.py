import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

## Whole Preprocessing is done to achieve processed dataset
'''
data = pd.read_csv("NCERT_dataset.csv")
data=data.drop(["Topic","Explanation","Answer","Prerequisites","grade","Difficulty","StudentLevel","QuestionType","QuestionComplexity","EstimatedTime"],axis=1)

onehot_enc=OneHotEncoder()
subject_encoded=onehot_enc.fit_transform(data[["subject"]])
subject_encoded_df=pd.DataFrame(subject_encoded.toarray(),columns=onehot_enc.get_feature_names_out(["subject"]))
data=pd.concat([data.drop("subject",axis=1),subject_encoded_df],axis=1)

data=data.drop([ 'subject_Accountancy', 'subject_Biology',
       'subject_Business Studies','subject_Economics',
       'subject_Geography','subject_History', 'subject_Psychology','subject_Science',
       'subject_Social Studies', 'subject_Socialogy','subject_Political Science'],axis=1)


mask = (data["subject_Chemistry"] == 1) | (data["subject_Physics"] == 1)
data = data[mask]
'''
data=pd.read_csv("NCERT_processed_dataset.csv")
x=data["Question"]
y=data[["subject_Chemistry","subject_Physics"]].values
y = np.argmax(y, axis=1) 

tokenizer=Tokenizer()
tokenizer.fit_on_texts(x)
x_seq=tokenizer.texts_to_sequences(x)
x_padded=pad_sequences(x_seq,maxlen=100,padding="post")

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_padded,y,test_size=0.2,random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,GlobalAveragePooling1D
vocab_size = len(tokenizer.word_index) + 1

model=Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=64,input_length=100))
model.add(GlobalAveragePooling1D())
model.add(Dense(64,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer="adam",metrics=["accuracy"],loss="binary_crossentropy")

earlystopping=EarlyStopping(monitor="loss",patience=15,restore_best_weights=True)

model.fit(x_train,y_train,epochs=50,validation_data=(x_test,y_test),callbacks=earlystopping)
loss,acc=model.evaluate(x_test,y_test)
print(f"Test Accuracy : {acc:.2f}")

model.save("topic_classification.h5")

from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w") as f:
    f.write(tokenizer_json)












