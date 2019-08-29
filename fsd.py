import warnings
warnings.filterwarnings('ignore')


# Cloning Git Repo
! git clone https://github.com/adityasamant/Frog-Species-Detection

import pandas as pd
import numpy as np
data = pd.read_csv("Frogs_MFCCs.csv")
data = data.drop(["RecordID"],axis = 1)

X = data.drop(["Family","Genus","Species"],axis=1)
y = data[["Family","Genus","Species"]]

X

y

from sklearn.model_selection import train_test_split
trainX,testX,trainy,testy = train_test_split(X,y,train_size=0.7,random_state=17,stratify=y)
train = pd.concat([trainX,trainy],axis=1)
test = pd.concat([testX,testy],axis=1)

trainX

testX

trainy

testy

train

test

def hamloss(y_true,y_pred):
    return np.average(np.not_equal(y_true, y_pred))

def exactmatch(y_true,y_pred):
    return np.mean(np.all(np.equal(y_true, y_pred),axis=1))

def hamscore(y_true,y_pred):
    return np.average(np.equal(y_true, y_pred))

def hamdistance(y_true,y_pred):
    return np.sum(np.sum(np.not_equal(y_true, y_pred),axis = 1))/len(y_true)

def hamlossL(y_true,y_pred):
    return np.mean(np.not_equal(y_true, y_pred))

def exactmatchL(y_true,y_pred):
    return np.mean(np.equal(y_true, y_pred))

def hamscoreL(y_true,y_pred):
    return np.mean(np.equal(y_true, y_pred))

def hamdistanceL(y_true,y_pred):
    return np.mean(np.not_equal(y_true, y_pred))


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier

C_range = np.logspace(-2, 4, 7)
g_range = np.logspace(-9, 3, 13)

bestcg = []
for c in C_range:
    for g in g_range:
        print("*")
        bii_model = MultiOutputClassifier(OneVsRestClassifier(SVC(C=c,gamma=g),n_jobs=-1),n_jobs=-1)
        print(c,g)
        score = cross_val_score(bii_model,trainX,trainy,cv=10,n_jobs=-1)
        print(score.mean())
        bestcg.append([c,g,score.mean()])

bestCG = pd.DataFrame(bestcg,columns=['c','g','score'])
print("Best score is",max(bestCG["score"]))
bestCG.loc[bestCG["score"]==max(bestCG["score"])]

# transformation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
StrainX = sc.fit_transform(trainX)
StestX = sc.transform(testX)

# training and fitting model
bii_model = MultiOutputClassifier(OneVsRestClassifier(SVC(C=100.0,gamma=1.0),n_jobs=-1),n_jobs=-1)
bii_Smodel = MultiOutputClassifier(OneVsRestClassifier(SVC(C=100.0,gamma=1.0),n_jobs=-1),n_jobs=-1)
bii_model.fit(trainX,trainy)
bii_Smodel.fit(StrainX,trainy)

# predicting
pred = bii_model.predict(testX)
Spred = bii_Smodel.predict(StestX)

print("Hamming Loss",hamloss(testy,pred))
hamlossL(testy,pred)

print("Exact Match Score",exactmatch(testy, pred))
exactmatchL(testy, pred)

print("Standardized data Hamming Loss",hamloss(testy,Spred))
hamlossL(testy,Spred)

print("Standardized data Exact Match Score",exactmatch(testy, Spred))
exactmatchL(testy, Spred)


from sklearn.svm import LinearSVC

bestc = []
c = 0.00001
while c < 10003:
    print("*")
    biii_model = MultiOutputClassifier(LinearSVC(penalty='l1',C=c,dual=False),n_jobs=-1)
    print(c)
    score = cross_val_score(biii_model,StrainX,trainy,cv=10,n_jobs=-1)
    print(score.mean())
    bestc.append(score.mean())
    c = c * 10

biii_model = MultiOutputClassifier(LinearSVC(penalty='l1',C=1,dual=False),n_jobs=-1)
biii_model.fit(StrainX,trainy)
predl1 = biii_model.predict(StestX)
print("Test Accuracy is",biii_model.score(StestX,testy))

print("Hamming Loss",hamloss(testy, predl1))
hamlossL(testy, predl1)

print("Exact Match Score",exactmatch(testy, predl1))
exactmatchL(testy, predl1)


bestd = []
c = 0.00001
while c < 10003:
    print("*")
    biv_model = MultiOutputClassifier(LinearSVC(penalty='l1',C=c,dual=False,class_weight='balanced'),n_jobs=-1)
    print(c)
    score = cross_val_score(biv_model,StrainX,trainy,cv=10,n_jobs=-1)
    print(score.mean())
    bestd.append(score.mean())
    c = c * 10

biv_model = MultiOutputClassifier(LinearSVC(penalty='l1',C=1,dual=False,class_weight='balanced'),n_jobs=-1)
biv_model.fit(StrainX,trainy)
preds = biv_model.predict(StestX)
print("Test Accuracy is",biii_model.score(StestX,testy))

print("Hamming Loss",hamloss(testy, preds))
hamlossL(testy, preds)

print("Exact Match Score",exactmatch(testy, preds))
exactmatchL(testy, preds)


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

HD = []
for i in range(1,51):
    print("\nIteration",i)
    
    # (a)
    bestk = []
    for k in range(2,51):
        kmeans = KMeans(n_clusters=k, random_state=np.random.randint(1000))
        pred = kmeans.fit_predict(X)
        score = silhouette_score(X, pred)
        bestk.append(score)
    
    K = bestk.index(max(bestk))+2
    print("Best K is",K,"with silhouette score",max(bestk))
    kmean = KMeans(n_clusters=K, random_state=np.random.randint(1000)).fit_predict(X)
    temp = y.copy()
    temp['cluster'] = kmean
    
    # (b)
    fgs_c = []
    for cluster in range(K):
        famlist = temp.loc[temp['cluster']==cluster].Family.tolist()
        genlist = temp.loc[temp['cluster']==cluster].Genus.tolist()
        spelist = temp.loc[temp['cluster']==cluster].Species.tolist()
        fgs_c.append([max(famlist,key=famlist.count),max(genlist,key=genlist.count),max(spelist,key=spelist.count)])    
    fgs_C = pd.DataFrame(fgs_c,columns=temp.columns[:3])
    print(fgs_C)
    
    # (c)
    pred = []
    for i in kmean:
        pred.append(fgs_C.iloc[i].tolist())
        
    hl = hamloss(y, pred)
    hs = hamscore(y, pred)
    hd = hamdistance(y, pred)
    HD.append(hd)
    print("Hamming Loss:",hl)
    print(hamlossL(y, pred))
    print("Hamming Score:",hs)
    print(hamscoreL(y, pred))
    print("Hamming Distance:",hd)
    print(hamdistanceL(y, pred))

HDarray = np.array(HD)
print("Average of 50 Hamming Distance is",HDarray.mean())
print("Standard Deviation of 50 Hamming Distance is",HDarray.std())
